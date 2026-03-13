#ifndef FWMC_LOG_H_
#define FWMC_LOG_H_

#include <chrono>
#include <cstdarg>
#include <cstdio>

namespace mechabrain {

enum class LogLevel { kDebug, kInfo, kWarn, kError };

namespace detail {

inline FILE*& LogSink() {
  static FILE* sink = stderr;
  return sink;
}

inline LogLevel& MinLevel() {
  static LogLevel level = LogLevel::kInfo;
  return level;
}

inline auto& StartTime() {
  static auto t = std::chrono::steady_clock::now();
  return t;
}

}  // namespace detail

inline void SetLogSink(FILE* f) { detail::LogSink() = f; }
inline void SetLogLevel(LogLevel level) { detail::MinLevel() = level; }

inline void Log(LogLevel level, const char* fmt, ...) {
  if (level < detail::MinLevel()) return;

  auto now = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - detail::StartTime()).count();

  const char* tag = "?";
  switch (level) {
    case LogLevel::kDebug: tag = "DBG"; break;
    case LogLevel::kInfo:  tag = "INF"; break;
    case LogLevel::kWarn:  tag = "WRN"; break;
    case LogLevel::kError: tag = "ERR"; break;
  }

  FILE* sink = detail::LogSink();
  fprintf(sink, "[fwmc %s %lld] ", tag, static_cast<long long>(ms));

  va_list args;
  va_start(args, fmt);
  vfprintf(sink, fmt, args);
  va_end(args);

  fprintf(sink, "\n");
}

}  // namespace mechabrain

#endif  // FWMC_LOG_H_
