// The SmolVLM2 tokenizer (the one SmolVLA tokenises its task with), without Python.
//
// It is a GPT-2 style byte-level BPE, as tokenizer.json describes it:
//   1. added tokens (<|im_start|>, ...) are cut out first and map to their ids;
//   2. pre-tokenizer Digits(individual_digits): every digit becomes a piece of its own;
//   3. pre-tokenizer ByteLevel: the GPT-2 regex splits the rest into words, then each byte
//      becomes one printable character (space -> 'Ġ', newline -> 'Ċ');
//   4. BPE: adjacent symbols are merged, lowest rank first, while a merge applies.
// No normaliser, no BOS/EOS. export.py writes vocab.txt, merges.txt and added.txt from tokenizer.json.
//
// The regex classes (\p{L}, \p{N}, \s) are exact for ASCII and approximated by Unicode ranges
// elsewhere; tests.txt holds strings (Vietnamese, emoji, digits) with the ids Python gives them.
#pragma once

#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
 public:
  struct Token {
    int id;
    std::string text;  // the bytes it stands for
    bool added;        // an added/special token rather than a BPE one
  };

  explicit Tokenizer(const std::string& directory);

  std::vector<Token> encode(const std::string& text) const;
  std::string bytes_of(int id) const;  // what a token id stands for, for display
  size_t vocab_size() const { return id_to_token_.size(); }

  // Encode every line of tests.txt and compare with Python's ids. Returns the number of mismatches.
  int self_test(const std::string& tests_path, std::ostream& log) const;

 private:
  std::vector<int> bpe(const std::string& piece) const;

  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_rank_;  // "a b" -> rank
  std::vector<std::pair<std::string, int>> added_;   // longest first
  std::unordered_map<int, std::string> added_by_id_;
  std::string byte_char_[256];                       // byte -> its printable character (UTF-8)
  std::unordered_map<std::string, unsigned char> char_byte_;
  mutable std::unordered_map<std::string, std::vector<int>> cache_;
};

std::string unescape(const std::string& s);  // \n \t \r \\ as written by export.py
std::string printable(const std::string& bytes);  // newlines and tabs shown as \n \t
