#ifndef PARABRAIN_SCOPED_TIMER_H_
#define PARABRAIN_SCOPED_TIMER_H_

#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>

namespace mechabrain {

// Lightweight scoped timer. Prints elapsed time on destruction.
//
// Usage:
//   {
//     ScopedTimer t("synapse propagation");
//     // ... code to time ...
//   }  // prints: [timer] synapse propagation: 3.14 ms
//
// For repeated measurements, use TimerRegistry to accumulate stats.
struct ScopedTimer {
  using Clock = std::chrono::steady_clock;

  std::string label;
  Clock::time_point start;
  bool silent = false;  // set true to suppress output

  explicit ScopedTimer(const std::string& label)
      : label(label), start(Clock::now()) {}

  ~ScopedTimer() {
    if (!silent) {
      double ms = elapsed_ms();
      fprintf(stderr, "[timer] %s: %.3f ms\n", label.c_str(), ms);
    }
  }

  double elapsed_ms() const {
    auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - start).count();
  }

  double elapsed_us() const {
    auto now = Clock::now();
    return std::chrono::duration<double, std::micro>(now - start).count();
  }
};

// Accumulates timing samples for named sections. Thread-safe.
//
// Usage:
//   TimerRegistry& reg = TimerRegistry::Global();
//
//   for (int i = 0; i < 1000; ++i) {
//     {
//       auto t = reg.Scope("step_neurons");
//       StepNeurons(...);
//     }
//     {
//       auto t = reg.Scope("propagate");
//       Propagate(...);
//     }
//   }
//   reg.Report();  // prints per-section stats
//
struct TimerRegistry {
  struct Stats {
    std::string name;
    std::vector<double> samples_ms;

    size_t count() const { return samples_ms.size(); }

    double total_ms() const {
      double s = 0.0;
      for (double v : samples_ms) s += v;
      return s;
    }

    double mean_ms() const {
      return count() > 0 ? total_ms() / count() : 0.0;
    }

    double median_ms() const {
      if (samples_ms.empty()) return 0.0;
      auto sorted = samples_ms;
      std::sort(sorted.begin(), sorted.end());
      size_t n = sorted.size();
      return (n % 2 == 0) ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0
                          : sorted[n/2];
    }

    double p95_ms() const {
      if (samples_ms.empty()) return 0.0;
      auto sorted = samples_ms;
      std::sort(sorted.begin(), sorted.end());
      return sorted[std::min(sorted.size() - 1,
                             static_cast<size_t>(sorted.size() * 0.95))];
    }

    double stddev_ms() const {
      if (count() < 2) return 0.0;
      double m = mean_ms();
      double sq = 0.0;
      for (double v : samples_ms) sq += (v - m) * (v - m);
      return std::sqrt(sq / (count() - 1));
    }
  };

  // RAII scope handle that records elapsed time on destruction.
  struct ScopeHandle {
    TimerRegistry* reg;
    std::string name;
    std::chrono::steady_clock::time_point start;

    ScopeHandle(TimerRegistry* r, const std::string& n)
        : reg(r), name(n), start(std::chrono::steady_clock::now()) {}

    ~ScopeHandle() {
      double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      reg->Record(name, ms);
    }

    ScopeHandle(ScopeHandle&& o) noexcept
        : reg(o.reg), name(std::move(o.name)), start(o.start) {
      o.reg = nullptr;
    }

    ScopeHandle(const ScopeHandle&) = delete;
    ScopeHandle& operator=(const ScopeHandle&) = delete;
    ScopeHandle& operator=(ScopeHandle&&) = delete;
  };

  ScopeHandle Scope(const std::string& name) {
    return ScopeHandle(this, name);
  }

  void Record(const std::string& name, double ms) {
    std::lock_guard<std::mutex> lock(mu_);
    sections_[name].samples_ms.push_back(ms);
  }

  // Print summary sorted by total time descending.
  void Report(FILE* out = stderr) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::pair<std::string, Stats>> sorted;
    for (auto& [k, v] : sections_) {
      Stats s;
      s.name = k;
      s.samples_ms = v.samples_ms;
      sorted.push_back({k, std::move(s)});
    }
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b) { return a.second.total_ms() > b.second.total_ms(); });

    fprintf(out, "\n[profiler] %-30s %8s %10s %10s %10s %10s\n",
            "section", "calls", "total_ms", "mean_ms", "p50_ms", "p95_ms");
    fprintf(out, "[profiler] %-30s %8s %10s %10s %10s %10s\n",
            "-------", "-----", "--------", "-------", "------", "------");
    for (auto& [name, s] : sorted) {
      fprintf(out, "[profiler] %-30s %8zu %10.2f %10.3f %10.3f %10.3f\n",
              name.c_str(), s.count(), s.total_ms(),
              s.mean_ms(), s.median_ms(), s.p95_ms());
    }
    fprintf(out, "\n");
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    sections_.clear();
  }

  // Global singleton for convenience.
  static TimerRegistry& Global() {
    static TimerRegistry instance;
    return instance;
  }

private:
  mutable std::mutex mu_;

  struct SectionData {
    std::vector<double> samples_ms;
  };
  std::unordered_map<std::string, SectionData> sections_;
};

}  // namespace mechabrain

#endif  // PARABRAIN_SCOPED_TIMER_H_
