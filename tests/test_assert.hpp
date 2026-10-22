#pragma once
// Minimal assertion helpers for the unit tests (no external framework).

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

namespace cstfem_test {
inline int failures = 0;
inline int checks = 0;
}  // namespace cstfem_test

#define CHECK(cond)                                                                            \
  do {                                                                                         \
    ++cstfem_test::checks;                                                                     \
    if (!(cond)) {                                                                             \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);            \
      ++cstfem_test::failures;                                                                 \
    }                                                                                          \
  } while (0)

// |a - b| <= tol
#define CHECK_NEAR(a, b, tol)                                                                  \
  do {                                                                                         \
    ++cstfem_test::checks;                                                                     \
    const double va_ = (a), vb_ = (b), vt_ = (tol);                                            \
    if (!(std::abs(va_ - vb_) <= vt_)) {                                                       \
      std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s = %.17g, %s = %.17g, tol %.3g\n",     \
                   __FILE__, __LINE__, #a, va_, #b, vb_, vt_);                                 \
      ++cstfem_test::failures;                                                                 \
    }                                                                                          \
  } while (0)

// `stmt` must throw a std::exception whose message contains `substr`.
#define CHECK_THROWS(stmt, substr)                                                             \
  do {                                                                                         \
    ++cstfem_test::checks;                                                                     \
    bool thrown_ = false;                                                                      \
    try {                                                                                      \
      stmt;                                                                                    \
    } catch (const std::exception& e_) {                                                       \
      thrown_ = true;                                                                          \
      if (std::string(e_.what()).find(substr) == std::string::npos) {                         \
        std::fprintf(stderr, "%s:%d: wrong exception message: '%s' (expected '%s')\n",         \
                     __FILE__, __LINE__, e_.what(), substr);                                   \
        ++cstfem_test::failures;                                                               \
      }                                                                                        \
    }                                                                                          \
    if (!thrown_) {                                                                            \
      std::fprintf(stderr, "%s:%d: expected exception from: %s\n", __FILE__, __LINE__, #stmt); \
      ++cstfem_test::failures;                                                                 \
    }                                                                                          \
  } while (0)

#define RUN_TEST(fn)                                                                           \
  do {                                                                                         \
    const int before_ = cstfem_test::failures;                                                 \
    fn();                                                                                      \
    std::printf("[%s] %s\n", cstfem_test::failures == before_ ? "PASS" : "FAIL", #fn);         \
  } while (0)

inline int test_summary() {
  std::printf("%d checks, %d failures\n", cstfem_test::checks, cstfem_test::failures);
  return cstfem_test::failures == 0 ? 0 : 1;
}
