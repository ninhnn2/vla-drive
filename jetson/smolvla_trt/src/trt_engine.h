// One TensorRT engine, with a device buffer for each of its inputs and outputs.
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

void cuda_check(cudaError_t status, const char* what);

class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(Severity level = Severity::kWARNING) : level_(level) {}
  void log(Severity severity, const char* msg) noexcept override;

 private:
  Severity level_;
};

struct TensorInfo {
  std::string name;
  bool is_input = false;
  nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
  std::vector<int64_t> shape;
  size_t bytes = 0;

  size_t count() const;
  std::string describe() const;  // "kv_cache  out  float32 [16, 2, 1, 113, 5, 64]"
};

class TrtEngine {
 public:
  TrtEngine(const std::string& path, nvinfer1::IRuntime& runtime);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  const std::string& name() const { return name_; }
  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo& info(const std::string& tensor) const;

  // The device buffer of a tensor: allocated here, unless share() gave it another one.
  void* device(const std::string& tensor) const;
  // Use `ptr` (another engine's buffer, same size) for `tensor`, so nothing is copied between them.
  void share(const std::string& tensor, void* ptr);

  void upload(const std::string& tensor, const void* host, cudaStream_t stream);
  void download(const std::string& tensor, void* host, cudaStream_t stream) const;

  // Run once; returns the GPU time in milliseconds (the stream is synchronised).
  float run(cudaStream_t stream);

 private:
  std::string name_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<TensorInfo> tensors_;
  std::map<std::string, void*> device_;
  std::vector<void*> owned_;
  cudaEvent_t start_ = nullptr, stop_ = nullptr;
};
