#pragma once
// Minimal JSON value formatting for timings.json (no external dependencies).

#include <cmath>
#include <cstdio>
#include <string>

namespace cstfem {

// Number with full precision; non-finite values become null.
inline std::string json_number(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

inline std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

}  // namespace cstfem
