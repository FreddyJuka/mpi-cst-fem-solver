#include "cstfem/case.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <ostream>
#include <stdexcept>

namespace cstfem {

namespace {

std::string trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r");
  if (first == std::string::npos) return {};
  const auto last = s.find_last_not_of(" \t\r");
  return s.substr(first, last - first + 1);
}

}  // namespace

CaseFile CaseFile::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error(path + ": cannot open file");
  return parse(in, path);
}

CaseFile CaseFile::parse(std::istream& in, const std::string& source_name) {
  CaseFile cf;
  cf.source_ = source_name;
  std::string raw;
  int line_no = 0;
  while (std::getline(in, raw)) {
    ++line_no;
    const std::string line = trim(raw.substr(0, raw.find('#')));
    if (line.empty()) continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos)
      throw std::runtime_error(cf.where(line_no) + ": expected 'key = value', got '" + line + "'");
    const std::string key = trim(line.substr(0, eq));
    const std::string value = trim(line.substr(eq + 1));
    if (key.empty()) throw std::runtime_error(cf.where(line_no) + ": empty key");
    if (key.find_first_of(" \t") != std::string::npos)
      throw std::runtime_error(cf.where(line_no) + ": key '" + key + "' contains whitespace");
    const auto [it, inserted] = cf.entries_.emplace(key, Entry{value, line_no});
    if (!inserted)
      throw std::runtime_error(cf.where(line_no) + ": duplicate key '" + key +
                               "' (first defined on line " + std::to_string(it->second.line) +
                               ")");
    cf.order_.push_back(key);
  }
  return cf;
}

bool CaseFile::has(const std::string& key) const { return entries_.count(key) != 0; }

std::string CaseFile::where(int line) const { return source_ + ":" + std::to_string(line); }

const CaseFile::Entry& CaseFile::entry(const std::string& key) const {
  const auto it = entries_.find(key);
  if (it == entries_.end())
    throw std::runtime_error(source_ + ": missing required key '" + key + "'");
  return it->second;
}

std::string CaseFile::get_string(const std::string& key) const { return entry(key).value; }

std::string CaseFile::get_string(const std::string& key, const std::string& fallback) const {
  return has(key) ? get_string(key) : fallback;
}

double CaseFile::get_double(const std::string& key) const {
  const Entry& e = entry(key);
  const char* begin = e.value.c_str();
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(begin, &end);
  if (e.value.empty() || end != begin + e.value.size())
    throw std::runtime_error(where(e.line) + ": key '" + key + "': '" + e.value +
                             "' is not a number");
  if (errno == ERANGE || !std::isfinite(v))
    throw std::runtime_error(where(e.line) + ": key '" + key + "': '" + e.value +
                             "' is out of range or not finite");
  return v;
}

double CaseFile::get_double(const std::string& key, double fallback) const {
  return has(key) ? get_double(key) : fallback;
}

int CaseFile::get_int(const std::string& key) const {
  const Entry& e = entry(key);
  const char* begin = e.value.c_str();
  char* end = nullptr;
  errno = 0;
  const long v = std::strtol(begin, &end, 10);
  if (e.value.empty() || end != begin + e.value.size())
    throw std::runtime_error(where(e.line) + ": key '" + key + "': '" + e.value +
                             "' is not an integer");
  if (errno == ERANGE || v < INT_MIN || v > INT_MAX)
    throw std::runtime_error(where(e.line) + ": key '" + key + "': '" + e.value +
                             "' is out of range");
  return static_cast<int>(v);
}

int CaseFile::get_int(const std::string& key, int fallback) const {
  return has(key) ? get_int(key) : fallback;
}

std::vector<std::string> CaseFile::unknown_keys(const std::vector<std::string>& known) const {
  std::vector<std::string> unknown;
  for (const auto& key : order_) {
    if (std::find(known.begin(), known.end(), key) == known.end()) unknown.push_back(key);
  }
  return unknown;
}

int CaseFile::warn_unknown_keys(const std::vector<std::string>& known, std::ostream& os) const {
  const auto unknown = unknown_keys(known);
  for (const auto& key : unknown)
    os << "warning: " << where(entries_.at(key).line) << ": unknown key '" << key << "'\n";
  return static_cast<int>(unknown.size());
}

}  // namespace cstfem
