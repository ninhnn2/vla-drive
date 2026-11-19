#include "smolvla.h"

#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

namespace {

double since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

float rms(const float* x, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; ++i) s += static_cast<double>(x[i]) * x[i];
  return static_cast<float>(std::sqrt(s / static_cast<double>(n)));
}

}  // namespace

SmolVLA::SmolVLA(const std::string& model_dir, const std::string& engine_dir)
    : model_dir_(model_dir),
      engine_dir_(engine_dir.empty() ? model_dir : engine_dir),
      meta_(model_dir + "/meta.txt"),
      tokenizer_(model_dir + "/tokenizer") {
  chunk_size = meta_.integer("chunk_size");
  max_action_dim = meta_.integer("max_action_dim");
  max_state_dim = meta_.integer("max_state_dim");
  num_image_tokens = meta_.integer("num_image_tokens");
  grid = meta_.integer("image_token_grid");
  max_lang_tokens = meta_.integer("max_lang_tokens");
  layers = meta_.integer("lm_layers");
  expert_hidden = meta_.integer("expert_hidden");
  pad_token_id = meta_.integer("pad_token_id");
  chunk_dt = meta_.number("chunk_dt_s");
  auto size = meta_.floats("image_size");
  image_w = static_cast<int>(size.at(0)), image_h = static_cast<int>(size.at(1));
  state_mean_ = meta_.floats("state_mean"), state_std_ = meta_.floats("state_std");
  action_mean_ = meta_.floats("action_mean"), action_std_ = meta_.floats("action_std");
  norm_eps_ = meta_.number("norm_eps");
  min_period_ = meta_.number("min_period"), max_period_ = meta_.number("max_period");
}

SmolVLA::~SmolVLA() {
  expert_.reset();
  prefix_.reset();
  vision_.reset();
  runtime_.reset();
  if (stream_) cudaStreamDestroy(stream_);
}

void SmolVLA::load_engines() {
  if (vision_) return;
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) throw std::runtime_error("cannot create the TensorRT runtime");
  vision_ = std::make_unique<TrtEngine>(engine_dir_ + "/vision.engine", *runtime_);
  prefix_ = std::make_unique<TrtEngine>(engine_dir_ + "/prefix.engine", *runtime_);
  expert_ = std::make_unique<TrtEngine>(engine_dir_ + "/expert.engine", *runtime_);
  // Hand each engine's output to the next one where it already lies: nothing is copied.
  prefix_->share("image_tokens", vision_->device("image_tokens"));
  expert_->share("kv_cache", prefix_->device("kv_cache"));
  cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
}

std::vector<const TrtEngine*> SmolVLA::engines() const { return {vision_.get(), prefix_.get(), expert_.get()}; }

std::vector<float> SmolVLA::noise(uint64_t seed) const {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> out(static_cast<size_t>(chunk_size) * max_action_dim);
  for (auto& v : out) v = normal(rng);
  return out;
}

// create_sinusoidal_pos_embedding(t, 720, 0.004, 4.0): LeRobot computes it in float64.
void SmolVLA::time_embedding(double t, std::vector<float>& out) const {
  const int half = expert_hidden / 2;
  out.resize(expert_hidden);
  for (int i = 0; i < half; ++i) {
    double fraction = half > 1 ? static_cast<double>(i) / (half - 1) : 0.0;
    double period = min_period_ * std::pow(max_period_ / min_period_, fraction);
    double angle = 2.0 * M_PI / period * t;
    out[i] = static_cast<float>(std::sin(angle));
    out[half + i] = static_cast<float>(std::cos(angle));
  }
}

Trace SmolVLA::infer(const Observation& obs, const std::vector<float>& noise, int num_steps) {
  load_engines();
  if (noise.size() != static_cast<size_t>(chunk_size) * max_action_dim) throw std::runtime_error("noise has the wrong size");
  if (num_steps < 1) throw std::runtime_error("at least one denoising step");
  Trace tr;
  const auto t0 = std::chrono::steady_clock::now();

  // 1. Text: append the newline, tokenise, truncate, pad on the right.
  auto t1 = std::chrono::steady_clock::now();
  tr.task = obs.task;
  if (tr.task.empty() || tr.task.back() != '\n') tr.task += '\n';
  tr.tokens = tokenizer_.encode(tr.task);
  tr.ids.assign(max_lang_tokens, pad_token_id);
  tr.mask.assign(max_lang_tokens, 0);
  for (size_t i = 0; i < tr.tokens.size() && i < static_cast<size_t>(max_lang_tokens); ++i)
    tr.ids[i] = tr.tokens[i].id, tr.mask[i] = 1;
  tr.tokenize_ms = since(t1);

  // 2. Image and state.
  t1 = std::chrono::steady_clock::now();
  tr.image = prepare_image(obs.rgb, image_w, image_h);
  tr.state[0] = obs.speed, tr.state[1] = obs.steer;
  std::vector<float> state(max_state_dim, 0.0f);
  for (int i = 0; i < 2; ++i) {
    tr.state_norm[i] = static_cast<float>((tr.state[i] - state_mean_[i]) / (state_std_[i] + norm_eps_));
    state[i] = tr.state_norm[i];
  }
  tr.prefix_mask.assign(num_image_tokens, 1);
  tr.prefix_mask.insert(tr.prefix_mask.end(), tr.mask.begin(), tr.mask.end());
  tr.prefix_mask.push_back(1);
  tr.preprocess_ms = since(t1);

  // 3. Vision: 512x512 -> 64 tokens.
  vision_->upload("pixel_values", tr.image.chw.data(), stream_);
  tr.vision_ms = vision_->run(stream_);
  tr.image_tokens.resize(vision_->info("image_tokens").count());
  vision_->download("image_tokens", tr.image_tokens.data(), stream_);
  const size_t width = tr.image_tokens.size() / num_image_tokens;
  for (int i = 0; i < num_image_tokens; ++i) tr.token_rms.push_back(rms(&tr.image_tokens[i * width], width));

  // 4. Prefix: image, text and state through the 16 language-model layers -> KV cache.
  prefix_->upload("lang_ids", tr.ids.data(), stream_);
  prefix_->upload("lang_mask", tr.mask.data(), stream_);
  prefix_->upload("state", state.data(), stream_);
  tr.prefix_ms = prefix_->run(stream_);
  {
    const auto& kv = prefix_->info("kv_cache");  // [layers, 2, 1, 113, 5, 64]
    std::vector<float> cache(kv.count());
    prefix_->download("kv_cache", cache.data(), stream_);
    const size_t per = cache.size() / (static_cast<size_t>(layers) * 2);
    for (int l = 0; l < layers; ++l) tr.kv_rms.push_back(rms(&cache[static_cast<size_t>(l) * 2 * per], per));
  }

  // 5. Denoise: Euler from t = 1 (noise) to t = 0, the expert giving the velocity at each step.
  expert_->upload("prefix_mask", tr.prefix_mask.data(), stream_);
  tr.noise = noise;
  std::vector<float> x = noise, v(x.size()), emb;
  const double dt = -1.0 / num_steps;
  for (int step = 0; step < num_steps; ++step) {
    const double t = 1.0 + step * dt;
    time_embedding(t, emb);
    expert_->upload("x_t", x.data(), stream_);
    expert_->upload("time_emb", emb.data(), stream_);
    float ms = expert_->run(stream_);
    expert_->download("v_t", v.data(), stream_);
    for (size_t i = 0; i < x.size(); ++i) x[i] += static_cast<float>(dt) * v[i];
    tr.expert_ms += ms;
    DenoiseStep s{static_cast<float>(t), rms(v.data(), v.size()), {0, 0}, ms};
    for (int k = 0; k < 2; ++k) s.first[k] = x[k] * action_std_[k] + action_mean_[k];
    tr.steps.push_back(s);
  }

  // 6. Unpad and unnormalise (LeRobot: x * std + mean, no eps).
  tr.chunk_norm = x;
  tr.chunk.resize(static_cast<size_t>(chunk_size) * 2);
  for (int i = 0; i < chunk_size; ++i)
    for (int k = 0; k < 2; ++k)
      tr.chunk[i * 2 + k] = x[static_cast<size_t>(i) * max_action_dim + k] * action_std_[k] + action_mean_[k];
  tr.total_ms = since(t0);
  return tr;
}
