#include "trt_engine.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= level_) std::cerr << "[TensorRT] " << msg << "\n";
}

static size_t element_size(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    default: throw std::runtime_error("unsupported TensorRT data type");
  }
}

static const char* type_name(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT: return "float32";
    case nvinfer1::DataType::kHALF: return "float16";
    case nvinfer1::DataType::kBF16: return "bfloat16";
    case nvinfer1::DataType::kINT32: return "int32";
    case nvinfer1::DataType::kINT64: return "int64";
    case nvinfer1::DataType::kINT8: return "int8";
    case nvinfer1::DataType::kUINT8: return "uint8";
    case nvinfer1::DataType::kBOOL: return "bool";
    default: return "?";
  }
}

size_t TensorInfo::count() const {
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(d);
  return n;
}

std::string TensorInfo::describe() const {
  std::ostringstream out;
  out << name << "  " << (is_input ? "in " : "out") << "  " << type_name(type) << " [";
  for (size_t i = 0; i < shape.size(); ++i) out << (i ? ", " : "") << shape[i];
  out << "]";
  return out.str();
}

TrtEngine::TrtEngine(const std::string& path, nvinfer1::IRuntime& runtime) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open engine " + path + " (build it with build_engines.sh)");
  std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  engine_.reset(runtime.deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) throw std::runtime_error("cannot load engine " + path + " (built with another TensorRT or GPU?)");
  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("cannot create an execution context for " + path);

  auto slash = path.find_last_of('/');
  name_ = path.substr(slash == std::string::npos ? 0 : slash + 1);

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    TensorInfo t;
    t.name = engine_->getIOTensorName(i);
    t.is_input = engine_->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    t.type = engine_->getTensorDataType(t.name.c_str());
    auto dims = engine_->getTensorShape(t.name.c_str());
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] < 0) throw std::runtime_error(name_ + ": " + t.name + " has a dynamic shape; export static shapes");
      t.shape.push_back(dims.d[d]);
    }
    t.bytes = t.count() * element_size(t.type);
    void* ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, t.bytes), "cudaMalloc");
    owned_.push_back(ptr);
    device_[t.name] = ptr;
    if (!context_->setTensorAddress(t.name.c_str(), ptr)) throw std::runtime_error("setTensorAddress " + t.name);
    tensors_.push_back(t);
  }
  cuda_check(cudaEventCreate(&start_), "cudaEventCreate");
  cuda_check(cudaEventCreate(&stop_), "cudaEventCreate");
}

TrtEngine::~TrtEngine() {
  context_.reset();
  engine_.reset();
  for (void* ptr : owned_) cudaFree(ptr);
  if (start_) cudaEventDestroy(start_);
  if (stop_) cudaEventDestroy(stop_);
}

const TensorInfo& TrtEngine::info(const std::string& tensor) const {
  for (const auto& t : tensors_)
    if (t.name == tensor) return t;
  throw std::runtime_error(name_ + " has no tensor named " + tensor);
}

void* TrtEngine::device(const std::string& tensor) const {
  info(tensor);
  return device_.at(tensor);
}

void TrtEngine::share(const std::string& tensor, void* ptr) {
  info(tensor);
  device_[tensor] = ptr;
  if (!context_->setTensorAddress(tensor.c_str(), ptr)) throw std::runtime_error("setTensorAddress " + tensor);
}

void TrtEngine::upload(const std::string& tensor, const void* host, cudaStream_t stream) {
  cuda_check(cudaMemcpyAsync(device(tensor), host, info(tensor).bytes, cudaMemcpyHostToDevice, stream), "upload");
}

void TrtEngine::download(const std::string& tensor, void* host, cudaStream_t stream) const {
  cuda_check(cudaMemcpyAsync(host, device(tensor), info(tensor).bytes, cudaMemcpyDeviceToHost, stream), "download");
  cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

float TrtEngine::run(cudaStream_t stream) {
  cuda_check(cudaEventRecord(start_, stream), "cudaEventRecord");
  if (!context_->enqueueV3(stream)) throw std::runtime_error(name_ + ": enqueueV3 failed");
  cuda_check(cudaEventRecord(stop_, stream), "cudaEventRecord");
  cuda_check(cudaEventSynchronize(stop_), "cudaEventSynchronize");
  float ms = 0.f;
  cuda_check(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
  return ms;
}
