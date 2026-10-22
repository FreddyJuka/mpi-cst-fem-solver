#pragma once

#include <istream>
#include <map>
#include <string>
#include <vector>

namespace cstfem {

// Plain `key = value` case file (SPEC section 3.2). Values are stored as raw strings;
// errors are std::runtime_error with "source:line: message".
class CaseFile {
public:
  static CaseFile load(const std::string& path);
  static CaseFile parse(std::istream& in, const std::string& source_name);

  bool has(const std::string& key) const;

  std::string get_string(const std::string& key) const;
  std::string get_string(const std::string& key, const std::string& fallback) const;
  double get_double(const std::string& key) const;
  double get_double(const std::string& key, double fallback) const;
  int get_int(const std::string& key) const;
  int get_int(const std::string& key, int fallback) const;

  // Keys present in the file but not in `known`, in file order.
  std::vector<std::string> unknown_keys(const std::vector<std::string>& known) const;
  // Prints one warning line per unknown key to `os`; returns the number of warnings.
  int warn_unknown_keys(const std::vector<std::string>& known, std::ostream& os) const;

  const std::string& source() const { return source_; }

private:
  struct Entry {
    std::string value;
    int line = 0;
  };
  const Entry& entry(const std::string& key) const;
  std::string where(int line) const;

  std::string source_;
  std::map<std::string, Entry> entries_;
  std::vector<std::string> order_;
};

}  // namespace cstfem
