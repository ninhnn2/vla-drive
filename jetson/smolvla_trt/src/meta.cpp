#include "meta.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

Meta::Meta(const std::string& path) : path_(path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("no " + path + " (run export.py, then copy its output here)");
  std::string line;
  while (std::getline(in, line)) {
    auto eq = line.find(" = ");
    if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
    values_[line.substr(0, eq)] = line.substr(eq + 3);
  }
}

const std::string& Meta::str(const std::string& key) const {
  auto it = values_.find(key);
  if (it == values_.end()) throw std::runtime_error(path_ + " has no " + key);
  return it->second;
}

int Meta::integer(const std::string& key) const { return std::stoi(str(key)); }
double Meta::number(const std::string& key) const { return std::stod(str(key)); }

std::vector<float> Meta::floats(const std::string& key) const {
  std::istringstream in(str(key));
  std::vector<float> out;
  for (float v; in >> v;) out.push_back(v);
  return out;
}

std::vector<std::string> Meta::words(const std::string& key) const {
  std::istringstream in(str(key));
  std::vector<std::string> out;
  for (std::string w; in >> w;) out.push_back(w);
  return out;
}

std::vector<float> read_floats(const std::string& path, size_t expected) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  auto bytes = static_cast<size_t>(in.tellg());
  if (bytes % sizeof(float) || (expected && bytes != expected * sizeof(float)))
    throw std::runtime_error(path + ": " + std::to_string(bytes) + " bytes, expected " +
                             std::to_string(expected * sizeof(float)));
  std::vector<float> out(bytes / sizeof(float));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
  return out;
}
