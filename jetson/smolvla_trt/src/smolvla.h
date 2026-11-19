// SmolVLA in three TensorRT engines, and everything around them done here in the open.
//
//   task text --tokenizer--> 48 ids + mask ---------------------------------+
//   camera frame --resize, pad, [-1,1]--> [3,512,512] --vision--> 64 tokens --+--prefix--> KV cache
//   [speed, steer] --(x - mean) / std, pad to 32--------------------------------+               |
//                                                                                              v
//   noise [16,32] --10 x (expert: v = f(x, t, KV); x += -0.1 * v)--> x --x * std + mean--> 16 x [steer, throttle]
#pragma once

#include "meta.h"
#include "preprocess.h"
#include "tokenizer.h"
#include "trt_engine.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

struct Observation {
  cv::Mat rgb;       // uint8 RGB camera frame, any size
  std::string task;  // the instruction; a newline is appended as LeRobot does
  float speed = 0, steer = 0;
};

struct DenoiseStep {
  float t;                 // flow-matching time: 1 is pure noise, 0 the action
  float v_rms;             // size of the velocity the expert predicts
  float first[2];          // the chunk's first action after this step, in robot units
  float ms;
};

struct Trace {
  // text
  std::string task;                           // as tokenised, with its newline
  std::vector<Tokenizer::Token> tokens;       // before truncation and padding
  std::vector<int32_t> ids, mask;             // [max_lang_tokens]
  // image
  ModelImage image;
  std::vector<float> image_tokens;            // [64 * 960]
  std::vector<float> token_rms;               // [64]: RMS of each image token
  // state
  float state[2], state_norm[2];
  // prefix and expert
  std::vector<int32_t> prefix_mask;           // [113]
  std::vector<float> kv_rms;                  // [layers]: RMS of each layer's keys
  std::vector<DenoiseStep> steps;
  std::vector<float> noise;                   // [16 * 32]
  std::vector<float> chunk_norm;              // [16 * 32] the final x, normalised and padded
  std::vector<float> chunk;                   // [16 * 2] steer, throttle in robot units
  // time, ms
  double tokenize_ms = 0, preprocess_ms = 0, vision_ms = 0, prefix_ms = 0, expert_ms = 0, total_ms = 0;
};

class SmolVLA {
 public:
  explicit SmolVLA(const std::string& model_dir, const std::string& engine_dir = "");
  ~SmolVLA();

  // Engines load on first use, so the tokenizer can be used without them.
  void load_engines();
  bool engines_loaded() const { return vision_ != nullptr; }

  Trace infer(const Observation& obs, const std::vector<float>& noise, int num_steps);
  std::vector<float> noise(uint64_t seed) const;  // N(0, 1), [chunk_size * max_action_dim]

  const Meta& meta() const { return meta_; }
  const std::string& model_dir() const { return model_dir_; }
  const Tokenizer& tokenizer() const { return tokenizer_; }
  std::vector<const TrtEngine*> engines() const;

  int chunk_size, max_action_dim, max_state_dim, num_image_tokens, grid, max_lang_tokens, layers, expert_hidden;
  int image_w, image_h, pad_token_id;
  double chunk_dt;

 private:
  void time_embedding(double t, std::vector<float>& out) const;

  std::string model_dir_, engine_dir_;
  Meta meta_;
  Tokenizer tokenizer_;
  std::vector<float> state_mean_, state_std_, action_mean_, action_std_;
  double norm_eps_, min_period_, max_period_;
  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<TrtEngine> vision_, prefix_, expert_;
  cudaStream_t stream_ = nullptr;
};
