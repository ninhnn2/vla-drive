// meta.txt (written by export.py): "key = value" lines, sizes and normalisation statistics.
#pragma once

#include <map>
#include <string>
#include <vector>

class Meta {
 public:
  explicit Meta(const std::string& path);
  bool has(const std::string& key) const { return values_.count(key) > 0; }
  const std::string& str(const std::string& key) const;
  int integer(const std::string& key) const;
  double number(const std::string& key) const;
  std::vector<float> floats(const std::string& key) const;
  std::vector<std::string> words(const std::string& key) const;

 private:
  std::map<std::string, std::string> values_;
  std::string path_;
};

// Raw float32 arrays as export.py writes them (numpy .tofile).
std::vector<float> read_floats(const std::string& path, size_t expected = 0);
