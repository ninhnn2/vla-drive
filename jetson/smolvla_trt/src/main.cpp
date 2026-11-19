// vla_probe: give the fine-tuned SmolVLA a camera frame, a task sentence and the car's state,
// and see what each stage does with them, on the Jetson, with TensorRT.
//
//   vla_probe --model DIR --image frame.png [--task "drive along the track"] [--speed 1.3 --steer 0]
//   vla_probe --model DIR --image frame.png --interactive      change the text, image, state; see the chunk move
//   vla_probe --model DIR --tokens "turn left"                 the tokenizer only (no engines needed)
//   vla_probe --model DIR --check                              against what Python computed (samples/)
#include "smolvla.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

struct Options {
  std::string model = ".", engines, image, task, noise_file, dump, tokens_only;
  float speed = 1.3f, steer = 0.0f;
  uint64_t seed = 0;
  int steps = 10, repeat = 0;
  bool interactive = false, check = false, quiet = false, verbose = false;
};

void usage() {
  std::cout <<
      "vla_probe: run SmolVLA (smolvla-v1) with TensorRT and show what each stage does\n\n"
      "  --model DIR       export.py's output: meta.txt, tokenizer/, samples/ (default .)\n"
      "  --engines DIR     where vision/prefix/expert.engine are (default: --model)\n"
      "  --image FILE      the camera frame (it was trained on the simulator's 320x240)\n"
      "  --task TEXT       the instruction (default: the one it was trained with)\n"
      "  --speed V         state: speed in m/s (default 1.3)\n"
      "  --steer S         state: wheel position in [-1, 1], + is left (default 0)\n"
      "  --seed N          seed of the starting noise (default 0); same seed, same noise\n"
      "  --noise FILE      float32 [16*32] starting noise instead\n"
      "  --steps N         denoising steps (default 10, as trained)\n"
      "  --repeat N        run N more times and report the timing\n"
      "  --dump DIR        write model_input.png, chunk.csv, trace.json\n"
      "  --interactive     then read commands: task/image/state/seed/steps/tokens/help/quit\n"
      "  --tokens TEXT     only tokenise TEXT (needs no engines)\n"
      "  --check           tokenizer and samples/ against Python; exit 1 on a mismatch\n"
      "  --quiet, -v       less, more\n";
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--model") o.model = next();
    else if (a == "--engines") o.engines = next();
    else if (a == "--image") o.image = next();
    else if (a == "--task") o.task = next();
    else if (a == "--speed") o.speed = std::stof(next());
    else if (a == "--steer") o.steer = std::stof(next());
    else if (a == "--seed") o.seed = std::stoull(next());
    else if (a == "--noise") o.noise_file = next();
    else if (a == "--steps") o.steps = std::stoi(next());
    else if (a == "--repeat") o.repeat = std::stoi(next());
    else if (a == "--dump") o.dump = next();
    else if (a == "--tokens") o.tokens_only = next();
    else if (a == "--interactive") o.interactive = true;
    else if (a == "--check") o.check = true;
    else if (a == "--quiet") o.quiet = true;
    else if (a == "-v" || a == "--verbose") o.verbose = true;
    else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    else throw std::runtime_error("unknown option " + a + " (--help)");
  }
  return o;
}

std::string rule(const std::string& title) {
  std::string line = "== " + title + " ";
  if (line.size() < 78) line += std::string(78 - line.size(), '=');
  return line;
}

// Steering as a bar: left of '|' is a left turn (steer > 0), right of it a right turn.
std::string steer_bar(float steer, int half = 12) {
  int n = static_cast<int>(std::lround(std::min(std::fabs(steer), 1.0f) * half));
  std::string left(half, ' '), right(half, ' ');
  if (steer > 0) std::fill(left.end() - n, left.end(), '#');
  else std::fill(right.begin(), right.begin() + n, '#');
  return "L" + left + "|" + right + "R";
}

cv::Mat load_rgb(const std::string& path) {
  cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) throw std::runtime_error("cannot read image " + path);
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

void print_tokens(const std::vector<Tokenizer::Token>& tokens, int max_tokens, bool all) {
  const int n = static_cast<int>(tokens.size());
  const int shown = all ? n : std::min(n, 24);
  for (int i = 0; i < shown; ++i) {
    std::cout << "    " << std::setw(3) << i << "  " << std::setw(6) << tokens[i].id << "  \""
              << printable(tokens[i].text) << "\"" << (tokens[i].added ? "  (special)" : "")
              << (i >= max_tokens ? "  -- cut: past the " + std::to_string(max_tokens) + "-token limit" : "") << "\n";
  }
  if (shown < n) std::cout << "    ... " << (n - shown) << " more (-v shows all)\n";
}

void report(const SmolVLA& m, const Trace& tr, const Options& o, const Trace* before) {
  const int n_text = std::accumulate(tr.mask.begin(), tr.mask.end(), 0);
  if (!o.quiet) {
    std::cout << rule("1. text") << "\n";
    std::cout << "  task \"" << printable(tr.task) << "\"  (LeRobot appends the newline)\n";
    std::cout << "  " << tr.tokens.size() << " tokens";
    if (static_cast<int>(tr.tokens.size()) > m.max_lang_tokens)
      std::cout << ", cut to " << m.max_lang_tokens;
    std::cout << " + " << (m.max_lang_tokens - n_text) << " padding (id " << m.pad_token_id
              << ", masked out) = " << m.max_lang_tokens << " text slots\n";
    print_tokens(tr.tokens, m.max_lang_tokens, o.verbose);

    const auto& im = tr.image;
    std::cout << rule("2. image") << "\n";
    std::cout << "  camera " << im.source_w << "x" << im.source_h << " -> resized " << im.resized_w << "x" << im.resized_h
              << " -> padded " << im.pad_left << " columns left, " << im.pad_top << " rows top -> " << im.width << "x"
              << im.height << ", values in [-1, 1]\n";
    std::cout << "  SigLIP cuts it into 32x32 patches of 16 px; the connector merges 4x4 of them into one token:\n"
              << "  " << m.num_image_tokens << " image tokens, " << m.grid << "x" << m.grid << ", each a "
              << im.width / m.grid << "x" << im.height / m.grid << " cell. RMS of each token (\"pad\": the cell is all padding):\n";
    std::cout << "        ";
    for (int c = 0; c < m.grid; ++c) std::cout << "   col" << c;
    std::cout << "\n";
    for (int r = 0; r < m.grid; ++r) {
      std::cout << "   row" << r << " ";
      for (int c = 0; c < m.grid; ++c) {
        const int cw = im.width / m.grid, ch = im.height / m.grid;
        bool padding = (r + 1) * ch <= im.pad_top || (c + 1) * cw <= im.pad_left;
        std::ostringstream cell;
        cell << std::fixed << std::setprecision(3) << tr.token_rms[r * m.grid + c];
        std::cout << std::setw(6) << (padding ? "pad" : cell.str()) << " ";
      }
      std::cout << "\n";
    }

    std::cout << rule("3. state") << "\n";
    std::cout << std::fixed << std::setprecision(3) << "  speed " << tr.state[0] << " m/s -> " << tr.state_norm[0]
              << "    steer " << tr.state[1] << " -> " << tr.state_norm[1]
              << "    ((x - mean) / std of the training data, then zeros to " << m.max_state_dim << ")\n";

    std::cout << rule("4. prefix: what the language model reads") << "\n";
    std::cout << "  tokens 0-" << m.num_image_tokens - 1 << " image | " << m.num_image_tokens << "-"
              << m.num_image_tokens + n_text - 1 << " text (" << n_text << ") | " << m.num_image_tokens + n_text << "-"
              << m.num_image_tokens + m.max_lang_tokens - 1 << " padding, masked | "
              << m.num_image_tokens + m.max_lang_tokens << " state\n";
    std::cout << "  image and text attend to each other; the state token sees them; nothing sees the state.\n"
              << "  " << m.layers << " layers run once and keep their keys and values (the KV cache) for the expert.\n";
    if (o.verbose) {
      std::cout << "  RMS of the keys, per layer:";
      for (size_t l = 0; l < tr.kv_rms.size(); ++l) std::cout << (l % 8 ? " " : "\n    ") << std::setprecision(2) << tr.kv_rms[l];
      std::cout << "\n";
    }

    std::cout << rule("5. denoising: noise (t=1) -> actions (t=0)") << "\n";
    std::cout << "  x starts as Gaussian noise [16 x 32]; each step the expert reads x, t and the KV cache and\n"
              << "  predicts a velocity v; x += -" << std::setprecision(2) << 1.0 / tr.steps.size() << " * v.\n";
    std::cout << "   step     t   rms(v)   first action: steer  throttle      ms\n";
    for (size_t s = 0; s < tr.steps.size(); ++s) {
      const auto& st = tr.steps[s];
      std::cout << "   " << std::setw(4) << s + 1 << std::setw(6) << std::setprecision(2) << st.t << std::setw(9)
                << std::setprecision(3) << st.v_rms << std::setw(21) << st.first[0] << std::setw(10) << st.first[1]
                << std::setw(8) << std::setprecision(2) << st.ms << "\n";
    }
  }

  std::cout << rule("6. the chunk: " + std::to_string(m.chunk_size) + " actions, " +
                    std::to_string(static_cast<int>(std::lround(m.chunk_dt * 1000))) + " ms apart") << "\n";
  std::cout << "    t (s)   steer  throttle   " << std::string(5, ' ') << "steering (+ is left)"
            << (before ? "      change from the last run" : "") << "\n";
  for (int i = 0; i < m.chunk_size; ++i) {
    float s = tr.chunk[i * 2], th = tr.chunk[i * 2 + 1];
    std::cout << std::fixed << std::setprecision(1) << "    " << std::setw(5) << i * m.chunk_dt << std::setprecision(3)
              << std::setw(8) << s << std::setw(10) << th << "   " << steer_bar(s);
    if (before)
      std::cout << std::showpos << std::setw(9) << s - before->chunk[i * 2] << std::setw(8) << th - before->chunk[i * 2 + 1]
                << std::noshowpos;
    std::cout << "\n";
  }
  float first4 = 0;
  for (int i = 0; i < 4; ++i) first4 += tr.chunk[i * 2] / 4;
  std::cout << "  over the first 0.4 s (what the car plays at ~125 ms per inference) it steers " << std::setprecision(3)
            << first4 << (first4 > 0.05 ? ": left" : first4 < -0.05 ? ": right" : ": about straight")
            << ". The drive stack clips to [-1, 1]; the supervisor may override.\n";

  std::cout << rule("timing") << "\n" << std::setprecision(2) << "  tokenise " << tr.tokenize_ms << " ms, image "
            << tr.preprocess_ms << " ms (CPU) | vision " << tr.vision_ms << " ms, prefix " << tr.prefix_ms
            << " ms, expert " << tr.steps.size() << " x " << tr.expert_ms / tr.steps.size() << " = " << tr.expert_ms
            << " ms (GPU) | total " << tr.total_ms << " ms\n";
}

void write_dump(const SmolVLA& m, const Trace& tr, const std::string& dir) {
  mkdir(dir.c_str(), 0755);
  cv::imwrite(dir + "/model_input.png", draw_token_grid(tr.image, m.grid));
  cv::imwrite(dir + "/token_rms.png", draw_token_grid(tr.image, m.grid, tr.token_rms));
  std::ofstream csv(dir + "/chunk.csv");
  csv << "t_s,steer,throttle\n";
  for (int i = 0; i < m.chunk_size; ++i) csv << i * m.chunk_dt << "," << tr.chunk[i * 2] << "," << tr.chunk[i * 2 + 1] << "\n";
  std::ofstream js(dir + "/trace.json");
  auto list = [&](const auto& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? ", " : "") << v[i];
    return o.str() + "]";
  };
  auto quote = [](const std::string& s) {
    std::string o = "\"";
    for (char c : s) o += c == '"' ? "\\\"" : c == '\\' ? "\\\\" : c == '\n' ? "\\n" : c == '\t' ? "\\t" : std::string(1, c);
    return o + "\"";
  };
  js << "{\n  \"task\": " << quote(tr.task) << ",\n  \"tokens\": [";
  for (size_t i = 0; i < tr.tokens.size(); ++i)
    js << (i ? ", " : "") << "{\"id\": " << tr.tokens[i].id << ", \"text\": " << quote(tr.tokens[i].text) << "}";
  js << "],\n  \"lang_ids\": " << list(tr.ids) << ",\n  \"lang_mask\": " << list(tr.mask)
     << ",\n  \"image\": {\"source\": [" << tr.image.source_w << ", " << tr.image.source_h << "], \"resized\": ["
     << tr.image.resized_w << ", " << tr.image.resized_h << "], \"pad_left\": " << tr.image.pad_left
     << ", \"pad_top\": " << tr.image.pad_top << "},\n  \"image_token_rms\": " << list(tr.token_rms)
     << ",\n  \"state\": [" << tr.state[0] << ", " << tr.state[1] << "], \"state_norm\": [" << tr.state_norm[0] << ", "
     << tr.state_norm[1] << "],\n  \"kv_key_rms\": " << list(tr.kv_rms) << ",\n  \"steps\": [";
  for (size_t s = 0; s < tr.steps.size(); ++s)
    js << (s ? ", " : "") << "{\"t\": " << tr.steps[s].t << ", \"v_rms\": " << tr.steps[s].v_rms << ", \"first\": ["
       << tr.steps[s].first[0] << ", " << tr.steps[s].first[1] << "], \"ms\": " << tr.steps[s].ms << "}";
  js << "],\n  \"chunk\": " << list(tr.chunk) << ",\n  \"ms\": {\"tokenize\": " << tr.tokenize_ms << ", \"image\": "
     << tr.preprocess_ms << ", \"vision\": " << tr.vision_ms << ", \"prefix\": " << tr.prefix_ms << ", \"expert\": "
     << tr.expert_ms << ", \"total\": " << tr.total_ms << "}\n}\n";
  std::cout << "  wrote " << dir << "/model_input.png, token_rms.png, chunk.csv, trace.json\n";
}

void timing(SmolVLA& m, const Observation& obs, const std::vector<float>& noise, int steps, int repeat) {
  std::vector<double> total, vision, prefix, expert;
  for (int i = 0; i < repeat; ++i) {
    auto tr = m.infer(obs, noise, steps);
    total.push_back(tr.total_ms), vision.push_back(tr.vision_ms), prefix.push_back(tr.prefix_ms), expert.push_back(tr.expert_ms);
  }
  auto stat = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    std::ostringstream o;
    o << std::fixed << std::setprecision(1) << "median " << v[v.size() / 2] << "  min " << v.front() << "  max " << v.back();
    return o.str();
  };
  std::cout << rule("timing over " + std::to_string(repeat) + " runs (ms)") << "\n"
            << "  vision  " << stat(vision) << "\n  prefix  " << stat(prefix) << "\n  expert  " << stat(expert)
            << "  (" << steps << " steps)\n  total   " << stat(total) << "  (with the CPU work and the copies)\n";
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, static_cast<double>(std::fabs(a[i] - b[i])));
  return d;
}

int check(SmolVLA& m) {
  std::cout << rule("check: the tokenizer against Python") << "\n";
  int failures = m.tokenizer().self_test(m.model_dir() + "/tokenizer/tests.txt", std::cout);
  m.load_engines();

  std::cout << rule("check: samples/ against Python") << "\n";
  const std::string dir = m.model_dir() + "/samples/";
  std::ifstream list(dir + "samples.txt");
  if (!list) throw std::runtime_error("no " + dir + "samples.txt");
  std::string line;
  std::cout << "  sample                    pixels  ids   image tokens      chunk vs fp32  vs bf16   teacher steer[0]  model\n";
  while (std::getline(list, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    std::string name, speed, steer;
    std::getline(row, name, '\t');
    std::string state_part, ids_part;
    std::getline(row, state_part, '\t');
    std::getline(row, ids_part);
    std::istringstream sp(state_part), ip(ids_part);
    Observation obs;
    sp >> obs.speed >> obs.steer;
    std::vector<int32_t> ids;
    for (int id; ip >> id;) ids.push_back(id);
    obs.rgb = load_rgb(dir + name + ".png");
    obs.task = m.meta().str("default_task");
    auto noise = read_floats(dir + name + ".noise.f32", static_cast<size_t>(m.chunk_size) * m.max_action_dim);
    auto tr = m.infer(obs, noise, 10);
    auto pixels = read_floats(dir + name + ".pixels.f32", tr.image.chw.size());
    auto tokens = read_floats(dir + name + ".tokens.f32", tr.image_tokens.size());
    auto fp32 = read_floats(dir + name + ".fp32.f32", tr.chunk.size());
    auto bf16 = read_floats(dir + name + ".bf16.f32", tr.chunk.size());
    auto plan = read_floats(dir + name + ".plan.f32", tr.chunk.size());
    double tok_scale = 0;
    for (float v : tokens) tok_scale = std::max(tok_scale, static_cast<double>(std::fabs(v)));
    double d_pix = max_diff(tr.image.chw, pixels), d_tok = max_diff(tr.image_tokens, tokens);
    double d32 = max_diff(tr.chunk, fp32), d16 = max_diff(tr.chunk, bf16);
    bool ids_ok = ids == tr.ids;
    std::cout << "  " << std::left << std::setw(24) << name << std::right << std::scientific << std::setprecision(1)
              << std::setw(8) << d_pix << "  " << (ids_ok ? "ok " : "BAD") << std::setw(9) << d_tok << " (of "
              << std::fixed << std::setprecision(0) << tok_scale << ")" << std::setprecision(3) << std::setw(12) << d32
              << std::setw(9) << d16 << std::setw(14) << plan[0] << std::setw(10) << tr.chunk[0] << "\n";
    // pixels: OpenCV and PyTorch resize alike to float rounding. chunk: FP16 against fp32; the bf16
    // LeRobot run is itself about 0.002 away from fp32, and steering differences under 0.02 do not show on the road.
    if (d_pix > 1e-3 || !ids_ok || d32 > 0.05) ++failures;
  }
  std::cout << "  (max |difference|; chunk in robot units, steer and throttle in [-1, 1])\n"
            << (failures ? "check: FAILED\n" : "check: ok\n");
  return failures ? 1 : 0;
}

void interactive(SmolVLA& m, Observation obs, Options o, std::vector<float> noise) {
  std::cout << "\ncommands: task <text> | image <file> | state <speed> <steer> | seed <n> | steps <n> | tokens <text>\n"
               "          dump <dir> | v (verbose on/off) | q (quiet on/off) | run | help | quit\n"
               "The noise stays the same until you change the seed, so a change in the chunk comes from your input.\n";
  Trace last = m.infer(obs, noise, o.steps);
  bool has_last = true;
  std::string line;
  while (std::cout << "\nvla> " << std::flush, std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    std::string rest;
    std::getline(in, rest);
    if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
    try {
      if (cmd.empty()) continue;
      if (cmd == "quit" || cmd == "exit") break;
      if (cmd == "help") { std::cout << "task <text> | image <file> | state <speed> <steer> | seed <n> | steps <n> | tokens <text> | dump <dir> | v | q | run | quit\n"; continue; }
      if (cmd == "tokens") { print_tokens(m.tokenizer().encode(unescape(rest)), m.max_lang_tokens, true); continue; }
      if (cmd == "dump") { write_dump(m, last, rest.empty() ? "dump" : rest); continue; }
      if (cmd == "task") obs.task = unescape(rest);
      else if (cmd == "image") obs.rgb = load_rgb(rest);
      else if (cmd == "state") { std::istringstream s(rest); s >> obs.speed >> obs.steer; }
      else if (cmd == "seed") { noise = m.noise(std::stoull(rest)); std::cout << "  new noise: seed " << rest << "\n"; }
      else if (cmd == "steps") o.steps = std::stoi(rest);
      else if (cmd == "v") { o.verbose = !o.verbose; std::cout << "  verbose " << (o.verbose ? "on" : "off") << "\n"; continue; }
      else if (cmd == "q") { o.quiet = !o.quiet; std::cout << "  quiet " << (o.quiet ? "on" : "off") << "\n"; continue; }
      else if (cmd != "run") { std::cout << "  unknown command (help)\n"; continue; }
      Trace tr = m.infer(obs, noise, o.steps);
      report(m, tr, o, has_last ? &last : nullptr);
      last = tr;
    } catch (const std::exception& e) {
      std::cout << "  error: " << e.what() << "\n";
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options o = parse(argc, argv);
    SmolVLA m(o.model, o.engines);
    std::cout << "SmolVLA " << m.meta().str("vlm") << ", fine-tuned; tokenizer " << m.tokenizer().vocab_size()
              << " tokens; chunk " << m.chunk_size << " x " << m.chunk_dt << " s\n";

    if (!o.tokens_only.empty()) {
      auto tokens = m.tokenizer().encode(unescape(o.tokens_only));
      std::cout << tokens.size() << " tokens for \"" << printable(unescape(o.tokens_only)) << "\":\n";
      print_tokens(tokens, m.max_lang_tokens, true);
      return 0;
    }
    if (o.check) return check(m);
    m.load_engines();
    if (!o.quiet)
      for (const auto* e : m.engines()) {
        std::cout << "  " << e->name() << ":";
        for (const auto& t : e->tensors()) std::cout << "  " << t.describe() << ";";
        std::cout << "\n";
      }

    if (o.image.empty()) throw std::runtime_error("--image is needed (or --check, --tokens); --help");
    Observation obs;
    obs.rgb = load_rgb(o.image);
    obs.task = o.task.empty() ? m.meta().str("default_task") : unescape(o.task);
    obs.speed = o.speed, obs.steer = o.steer;
    auto noise = o.noise_file.empty() ? m.noise(o.seed)
                                      : read_floats(o.noise_file, static_cast<size_t>(m.chunk_size) * m.max_action_dim);
    m.infer(obs, noise, o.steps);  // warm-up: the first run pays for lazy CUDA initialisation
    Trace tr = m.infer(obs, noise, o.steps);
    report(m, tr, o, nullptr);
    if (!o.dump.empty()) write_dump(m, tr, o.dump);
    if (o.repeat > 0) timing(m, obs, noise, o.steps, o.repeat);
    if (o.interactive) interactive(m, obs, o, noise);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "vla_probe: " << e.what() << "\n";
    return 1;
  }
}
