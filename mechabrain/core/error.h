#ifndef FWMC_ERROR_H_
#define FWMC_ERROR_H_

#include <string>

// std::expected requires C++23 library support. Clang with libstdc++ may not
// have it even with -std=c++23, so we detect via __cpp_lib_expected and fall
// back to a minimal variant-based polyfill.
#ifdef __has_include
#if __has_include(<expected>)
#include <expected>
#endif
#endif

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
#define FWMC_HAS_STD_EXPECTED 1
#else
#define FWMC_HAS_STD_EXPECTED 0
#include <variant>
#endif

namespace mechabrain {

enum class ErrorCode {
  kFileNotFound,
  kCorruptedData,
  kOutOfBounds,
  kInvalidParam,
};

struct Error {
  ErrorCode code;
  std::string message;
};

#if FWMC_HAS_STD_EXPECTED

template <typename T>
using Result = std::expected<T, Error>;

inline std::unexpected<Error> MakeError(ErrorCode code, std::string msg) {
  return std::unexpected<Error>(Error{code, std::move(msg)});
}

#else

// Minimal Result<T> polyfill using std::variant
template <typename T>
class Result {
 public:
  Result(T val) : data_(std::in_place_index<0>, std::move(val)) {}
  Result(Error err) : data_(std::in_place_index<1>, std::move(err)) {}

  bool has_value() const { return data_.index() == 0; }
  explicit operator bool() const { return has_value(); }
  T& value() { return std::get<0>(data_); }
  const T& value() const { return std::get<0>(data_); }
  T& operator*() { return value(); }
  const T& operator*() const { return value(); }
  Error& error() { return std::get<1>(data_); }
  const Error& error() const { return std::get<1>(data_); }

 private:
  std::variant<T, Error> data_;
};

// Returns Error directly; implicitly converts to any Result<T>
inline Error MakeError(ErrorCode code, std::string msg) {
  return Error{code, std::move(msg)};
}

#endif

}  // namespace mechabrain

#endif  // FWMC_ERROR_H_
