#ifndef PARABRAIN_MEMORY_TRACKER_H_
#define PARABRAIN_MEMORY_TRACKER_H_

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>

namespace mechabrain {

// Lightweight memory accounting for tracking allocation sizes.
//
// This does NOT hook malloc/free. Instead, call Track() when you allocate
// a known-size buffer and Untrack() when you free it. Designed for
// tracking the big allocations (neuron arrays, synapse tables, etc.)
// rather than every small allocation.
//
// Usage:
//   MemoryTracker& mem = MemoryTracker::Global();
//   neurons.Resize(100000);
//   mem.Track("neurons.v", 100000 * sizeof(float));
//   mem.Track("neurons.u", 100000 * sizeof(float));
//   mem.Track("synapses.csr", synapse_bytes);
//   mem.Report();
//
struct MemoryTracker {
  struct Entry {
    std::string name;
    size_t bytes = 0;
    std::string category;  // for grouping (e.g. "neurons", "synapses")
  };

  void Track(const std::string& name, size_t bytes,
             const std::string& category = "") {
    entries_[name] = {name, bytes, category};
    peak_bytes_ = std::max(peak_bytes_, TotalBytes());
  }

  void Untrack(const std::string& name) {
    entries_.erase(name);
  }

  size_t TotalBytes() const {
    size_t total = 0;
    for (auto& [_, e] : entries_) total += e.bytes;
    return total;
  }

  size_t PeakBytes() const { return peak_bytes_; }

  // Get bytes for a category.
  size_t CategoryBytes(const std::string& category) const {
    size_t total = 0;
    for (auto& [_, e] : entries_) {
      if (e.category == category) total += e.bytes;
    }
    return total;
  }

  void Report(FILE* out = stderr) const {
    // Sort by size descending
    std::vector<Entry> sorted;
    sorted.reserve(entries_.size());
    for (auto& [_, e] : entries_) sorted.push_back(e);
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b) { return a.bytes > b.bytes; });

    fprintf(out, "\n[memory] %-35s %12s  %s\n", "allocation", "size", "category");
    fprintf(out, "[memory] %-35s %12s  %s\n", "----------", "----", "--------");

    for (auto& e : sorted) {
      fprintf(out, "[memory] %-35s %12s  %s\n",
              e.name.c_str(), FormatBytes(e.bytes).c_str(),
              e.category.c_str());
    }

    fprintf(out, "[memory] %-35s %12s\n", "TOTAL", FormatBytes(TotalBytes()).c_str());
    fprintf(out, "[memory] %-35s %12s\n", "PEAK", FormatBytes(peak_bytes_).c_str());
    fprintf(out, "\n");
  }

  void Clear() {
    entries_.clear();
    peak_bytes_ = 0;
  }

  static MemoryTracker& Global() {
    static MemoryTracker instance;
    return instance;
  }

  // Helper: track a std::vector's memory.
  template <typename T>
  void TrackVector(const std::string& name, const std::vector<T>& vec,
                   const std::string& category = "") {
    Track(name, vec.capacity() * sizeof(T), category);
  }

private:
  std::unordered_map<std::string, Entry> entries_;
  size_t peak_bytes_ = 0;

  static std::string FormatBytes(size_t bytes) {
    char buf[32];
    if (bytes < 1024)
      snprintf(buf, sizeof(buf), "%zu B", bytes);
    else if (bytes < 1024 * 1024)
      snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else if (bytes < 1024ULL * 1024 * 1024)
      snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024));
    else
      snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024 * 1024));
    return buf;
  }
};

// RAII helper: tracks on construction, untracks on destruction.
struct ScopedAllocation {
  MemoryTracker& tracker;
  std::string name;

  ScopedAllocation(MemoryTracker& t, const std::string& n, size_t bytes,
                   const std::string& category = "")
      : tracker(t), name(n) {
    tracker.Track(name, bytes, category);
  }

  ~ScopedAllocation() {
    tracker.Untrack(name);
  }

  ScopedAllocation(const ScopedAllocation&) = delete;
  ScopedAllocation& operator=(const ScopedAllocation&) = delete;
};

}  // namespace mechabrain

#endif  // PARABRAIN_MEMORY_TRACKER_H_
