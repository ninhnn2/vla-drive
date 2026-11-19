#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string utf8(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

struct CodePoint {
  uint32_t cp;
  size_t begin, end;  // byte offsets in the text
};

std::vector<CodePoint> decode(const std::string& s) {
  std::vector<CodePoint> out;
  size_t i = 0;
  while (i < s.size()) {
    auto c = static_cast<unsigned char>(s[i]);
    int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + len > s.size()) len = 1;
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    out.push_back({cp, i, i + len});
    i += len;
  }
  return out;
}

bool in(uint32_t c, uint32_t lo, uint32_t hi) { return c >= lo && c <= hi; }

// \s in the regex: Unicode White_Space.
bool is_space(uint32_t c) {
  return in(c, 0x09, 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 || in(c, 0x2000, 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// \p{N}: numbers (also what the Digits pre-tokenizer isolates).
bool is_number(uint32_t c) {
  if (c < 0x80) return in(c, '0', '9');
  return c == 0xB2 || c == 0xB3 || c == 0xB9 || in(c, 0xBC, 0xBE) || in(c, 0x660, 0x669) || in(c, 0x6F0, 0x6F9) ||
         in(c, 0x966, 0x96F) || in(c, 0x2070, 0x2079) || in(c, 0x2080, 0x2089) || in(c, 0x2150, 0x2189) ||
         in(c, 0x2460, 0x249B) || in(c, 0x24EA, 0x24FF) || in(c, 0x2776, 0x2793) || in(c, 0xFF10, 0xFF19) ||
         in(c, 0x3007, 0x3007) || in(c, 0x3021, 0x3029);
}

// \p{L}: letters. Outside ASCII, everything that is not a known space, number, mark,
// punctuation or symbol block counts as a letter (Latin with diacritics, Greek, Cyrillic, CJK, ...).
bool is_letter(uint32_t c) {
  if (c < 0x80) return in(c, 'a', 'z') || in(c, 'A', 'Z');
  if (is_space(c) || is_number(c)) return false;
  if (in(c, 0x80, 0xBF)) return c == 0xAA || c == 0xB5 || c == 0xBA;  // Latin-1 punctuation and symbols
  if (c == 0xD7 || c == 0xF7) return false;                              // x and division signs
  if (in(c, 0x300, 0x36F) || in(c, 0x1AB0, 0x1AFF) || in(c, 0x1DC0, 0x1DFF) || in(c, 0x20D0, 0x20FF) ||
      in(c, 0xFE00, 0xFE0F) || in(c, 0xFE20, 0xFE2F))
    return false;                                                        // combining marks
  if (in(c, 0x2000, 0x2BFF)) return in(c, 0x2C00, 0x2BFF);               // punctuation, symbols, arrows, maths
  if (in(c, 0x3000, 0x303F)) return c == 0x3005 || c == 0x3006;          // CJK punctuation
  if (in(c, 0xFF00, 0xFF20) || in(c, 0xFF3B, 0xFF40) || in(c, 0xFF5B, 0xFF65)) return false;
  if (in(c, 0xE000, 0xF8FF) || in(c, 0x1F000, 0x1FAFF) || in(c, 0xF0000, 0x10FFFF)) return false;  // private, emoji
  return true;
}

// The GPT-2 pattern the ByteLevel pre-tokenizer uses:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
void gpt2_split(const std::vector<CodePoint>& cps, size_t from, size_t to, const std::string& text,
                std::vector<std::string>& out) {
  auto emit = [&](size_t a, size_t b) { out.push_back(text.substr(cps[a].begin, cps[b - 1].end - cps[a].begin)); };
  auto other = [](uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); };
  size_t i = from;
  while (i < to) {
    uint32_t c = cps[i].cp;
    if (c == '\'' && i + 1 < to) {
      uint32_t a = cps[i + 1].cp, b = i + 2 < to ? cps[i + 2].cp : 0;
      if (a == 's' || a == 't' || a == 'm' || a == 'd') { emit(i, i + 2); i += 2; continue; }
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) { emit(i, i + 3); i += 3; continue; }
    }
    size_t j = (c == ' ' && i + 1 < to) ? i + 1 : i;  // the optional leading space
    bool (*same)(uint32_t) = nullptr;
    if (is_letter(cps[j].cp)) same = is_letter;
    else if (is_number(cps[j].cp)) same = is_number;
    else if (other(cps[j].cp)) same = [](uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); };
    if (same) {
      size_t k = j;
      while (k < to && same(cps[k].cp)) ++k;
      emit(i, k);
      i = k;
      continue;
    }
    size_t k = i;  // whitespace: all of it at the end, else all but the last (which joins the next word)
    while (k < to && is_space(cps[k].cp)) ++k;
    size_t end = k == to ? k : (k - i > 1 ? k - 1 : i + 1);
    emit(i, end);
    i = end;
  }
}

}  // namespace

std::string unescape(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      char n = s[++i];
      out += n == 'n' ? '\n' : n == 't' ? '\t' : n == 'r' ? '\r' : n;
    } else {
      out += s[i];
    }
  }
  return out;
}

std::string printable(const std::string& bytes) {
  // Complete UTF-8 characters as they are; the bytes of a character split across tokens
  // (common for Vietnamese: "ậ" is 3 bytes, often in two tokens) as \xNN.
  std::string out;
  char hex[8];
  size_t i = 0;
  while (i < bytes.size()) {
    auto c = static_cast<unsigned char>(bytes[i]);
    size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
    bool whole = len > 0 && i + len <= bytes.size();
    for (size_t k = 1; whole && k < len; ++k) whole = (static_cast<unsigned char>(bytes[i + k]) >> 6) == 0x2;
    if (!whole) {
      std::snprintf(hex, sizeof hex, "\\x%02x", c);
      out += hex;
      ++i;
    } else if (c == '\n') { out += "\\n"; ++i; }
    else if (c == '\t') { out += "\\t"; ++i; }
    else if (c == '\r') { out += "\\r"; ++i; }
    else { out += bytes.substr(i, len); i += len; }
  }
  return out;
}

Tokenizer::Tokenizer(const std::string& directory) {
  // GPT-2's bytes_to_unicode: printable bytes stand for themselves, the rest are shifted past 255.
  int shifted = 0;
  for (int b = 0; b < 256; ++b) {
    bool keep = in(b, '!', '~') || in(b, 0xA1, 0xAC) || in(b, 0xAE, 0xFF);
    byte_char_[b] = utf8(keep ? b : 256 + shifted++);
    char_byte_[byte_char_[b]] = static_cast<unsigned char>(b);
  }

  std::ifstream vocab(directory + "/vocab.txt");
  if (!vocab) throw std::runtime_error("no " + directory + "/vocab.txt (run export.py)");
  std::string line;
  while (std::getline(vocab, line)) {
    vocab_[line] = static_cast<int>(id_to_token_.size());
    id_to_token_.push_back(line);
  }
  std::ifstream merges(directory + "/merges.txt");
  if (!merges) throw std::runtime_error("no " + directory + "/merges.txt");
  int rank = 0;
  while (std::getline(merges, line)) merge_rank_[line] = rank++;
  std::ifstream added(directory + "/added.txt");
  if (!added) throw std::runtime_error("no " + directory + "/added.txt");
  while (std::getline(added, line)) {
    auto tab = line.find('\t');
    int id = std::stoi(line.substr(0, tab));
    std::string content = unescape(line.substr(tab + 1));
    added_.push_back({content, id});
    added_by_id_[id] = content;
  }
  std::sort(added_.begin(), added_.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
}

std::vector<int> Tokenizer::bpe(const std::string& piece) const {
  auto hit = cache_.find(piece);
  if (hit != cache_.end()) return hit->second;
  std::vector<std::string> symbols;
  for (unsigned char b : piece) symbols.push_back(byte_char_[b]);
  while (symbols.size() > 1) {
    int best = INT_MAX;
    size_t at = 0;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto r = merge_rank_.find(symbols[i] + " " + symbols[i + 1]);
      if (r != merge_rank_.end() && r->second < best) best = r->second, at = i;
    }
    if (best == INT_MAX) break;
    const std::string a = symbols[at], b = symbols[at + 1];
    std::vector<std::string> merged;
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (i + 1 < symbols.size() && symbols[i] == a && symbols[i + 1] == b) {
        merged.push_back(a + b);
        ++i;
      } else {
        merged.push_back(symbols[i]);
      }
    }
    symbols.swap(merged);
  }
  std::vector<int> ids;
  for (const auto& s : symbols) {
    auto v = vocab_.find(s);
    if (v == vocab_.end()) throw std::runtime_error("BPE produced a symbol not in the vocabulary: " + s);
    ids.push_back(v->second);
  }
  cache_[piece] = ids;
  return ids;
}

std::vector<Tokenizer::Token> Tokenizer::encode(const std::string& text) const {
  std::vector<Token> out;
  auto encode_plain = [&](const std::string& part) {
    auto cps = decode(part);
    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < cps.size()) {  // Digits(individual_digits=true): each digit alone, then the GPT-2 regex
      if (is_number(cps[i].cp)) {
        pieces.push_back(part.substr(cps[i].begin, cps[i].end - cps[i].begin));
        ++i;
        continue;
      }
      size_t j = i;
      while (j < cps.size() && !is_number(cps[j].cp)) ++j;
      gpt2_split(cps, i, j, part, pieces);
      i = j;
    }
    for (const auto& piece : pieces)
      for (int id : bpe(piece)) out.push_back({id, bytes_of(id), false});
  };
  size_t start = 0, pos = 0;
  while (pos < text.size()) {
    const std::pair<std::string, int>* match = nullptr;
    for (const auto& a : added_)
      if (text.compare(pos, a.first.size(), a.first) == 0) { match = &a; break; }  // longest first
    if (!match) { ++pos; continue; }
    if (pos > start) encode_plain(text.substr(start, pos - start));
    out.push_back({match->second, match->first, true});
    pos += match->first.size();
    start = pos;
  }
  if (start < text.size()) encode_plain(text.substr(start));
  return out;
}

std::string Tokenizer::bytes_of(int id) const {
  auto a = added_by_id_.find(id);
  if (a != added_by_id_.end()) return a->second;
  if (id < 0 || id >= static_cast<int>(id_to_token_.size())) return "<?>";
  std::string bytes;
  for (const auto& cp : decode(id_to_token_[id])) {
    auto b = char_byte_.find(id_to_token_[id].substr(cp.begin, cp.end - cp.begin));
    bytes += b == char_byte_.end() ? '?' : static_cast<char>(b->second);
  }
  return bytes;
}

int Tokenizer::self_test(const std::string& tests_path, std::ostream& log) const {
  std::ifstream tests(tests_path);
  if (!tests) throw std::runtime_error("no " + tests_path);
  std::string line;
  int failures = 0, total = 0;
  while (std::getline(tests, line)) {
    auto tab = line.find('\t');
    std::string text = unescape(line.substr(0, tab));
    std::istringstream want_stream(line.substr(tab + 1));
    std::vector<int> want, got;
    for (int id; want_stream >> id;) want.push_back(id);
    for (const auto& t : encode(text)) got.push_back(t.id);
    ++total;
    if (got != want) {
      ++failures;
      log << "  FAIL \"" << printable(text) << "\"\n    python:";
      for (int id : want) log << ' ' << id;
      log << "\n    c++:   ";
      for (int id : got) log << ' ' << id;
      log << "\n";
    }
  }
  log << "  tokenizer: " << (total - failures) << "/" << total << " strings give Python's ids\n";
  return failures;
}
