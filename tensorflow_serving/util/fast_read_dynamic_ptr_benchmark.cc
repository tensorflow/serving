/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Benchmarks for read performance of the FastReadDynamicPtr class both with
// and without concurrent updates to the pointer being read. These simulate the
// common expected access pattern of FastReadDynamicPtr for systems with high
// read rates and low update rates.
//
// The main difference between this benchmark's and expected access patterns is
// that the reads simply repeat as quickly as possible with no time spent
// between reads doing useful work (e.g. performing some computation). This
// tests the maximum possible read contention. In real world usage, we can
// anticipate less contention as the system will be doing other useful work
// between reads from the FastReadDynamicPtr.
//
// Run with:
// bazel run -c opt \
// tensorflow_serving/util:fast_read_dynamic_ptr_benchmark --
// --benchmarks=.
// For a longer run time and more consistent results, consider a min time
// e.g.: --benchmark_min_time=60.0

#include <limits.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>

#include "tensorflow/contrib/batching/util/periodic_function.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/util/fast_read_dynamic_ptr.h"

namespace tensorflow {
namespace serving {

typedef FastReadDynamicPtr<int> FastReadIntPtr;

// This class maintains all state for a benchmark and handles the concurrency
// concerns around the concurrent read and update threads.
//
// Example:
//    BenchmarkState state(0 /* no updates */, false /* Don't do any work */);
//    state.Setup();
//    state.RunBenchmarkReadIterations(5 /* num_threads */, 42 /* iters */);
//    state.Teardown();
class BenchmarkState {
 public:
  BenchmarkState(const int update_micros, const bool do_work)
      : update_micros_(update_micros), do_work_(do_work) {}

  // Actually perform iters reads on the fast read ptr.
  void RunBenchmarkReadIterations(int num_threads, int iters);

  // Sets up the state for a benchmark run.
  // update_micros: Number of micros to sleep between updates. If set to 0, does
  //   not update at all after setup.
  void Setup();
  void Teardown();

 private:
  // Runs iters number of reads.
  void RunBenchmarkReads(int iters);

  // Runs continuously after setup and until teardown, with an optional sleep
  // for update_micros
  void RunUpdateThread();

  // To avoid having the benchmark timing include time spent scheduling threads,
  // we use this notification to notify when the read threads should begin.
  // This is notified immediately after the benchmark timing is started.
  Notification all_read_threads_scheduled_;

  // Store the update thread as it is only safe to complete teardown and
  // destruct state after it has exited.
  std::unique_ptr<PeriodicFunction> update_thread_;

  // The FastReadIntPtr being benchmarked primarily for read performance.
  FastReadIntPtr fast_ptr_;

  // The update interval in microseconds.
  int64 update_micros_;

  // In each iteration, to simulate a more realistic access pattern that does
  // more than content for the mutex.
  bool do_work_;
};

void BenchmarkState::RunUpdateThread() {
  int current_value;
  {
    std::shared_ptr<const int> current = fast_ptr_.get();
    current_value = *current;
  }
  std::unique_ptr<int> tmp(new int(current_value + 1));
  fast_ptr_.Update(std::move(tmp));
}

void BenchmarkState::Setup() {
  testing::StopTiming();

  // setup fast read int ptr:
  std::unique_ptr<int> i(new int(0));
  fast_ptr_.Update(std::move(i));

  if (update_micros_ > 0) {
    PeriodicFunction::Options pf_options;
    pf_options.thread_name_prefix =
        "FastReadDynamicPtr_Benchmark_Update_Thread";
    update_thread_.reset(new PeriodicFunction([this] { RunUpdateThread(); },
                                              update_micros_, pf_options));
  }

  testing::StartTiming();
}

void BenchmarkState::Teardown() {
  testing::StopTiming();

  // Destruct the update thread which blocks until it exits.
  update_thread_.reset();

  testing::StartTiming();
}

void BenchmarkState::RunBenchmarkReads(int iters) {
  // Wait until all_read_threads_scheduled_ has been notified.
  all_read_threads_scheduled_.WaitForNotification();

  for (int i = 0; i < iters; i++) {
    std::shared_ptr<const int> current = fast_ptr_.get();
    if (do_work_) {
      // Let's do some work, so that we are not just measuring contention in the
      // mutex.
      float count = 0;
      for (int i = 1; i < 10000; ++i) {
        count *= i;
      }
      CHECK_GE(count, 0);
    }
    // Prevent compiler optimizing this away.
    CHECK(*current != INT_MAX);
  }
}

void BenchmarkState::RunBenchmarkReadIterations(int num_threads, int iters) {
  testing::StopTiming();

  // The benchmarking system by default uses cpu time to calculate items per
  // second, which would include time spent by all the threads on the cpu.
  // Instead of that we use real-time here so that we can see items/s increasing
  // with increasing threads, which is easier to understand.
  testing::UseRealTime();
  testing::ItemsProcessed(num_threads * iters);

  thread::ThreadPool pool(Env::Default(), "RunBenchmarkReadThread",
                          num_threads);
  for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
    std::function<void()> run_reads_fn = [iters, this]() {
      RunBenchmarkReads(iters);
    };
    pool.Schedule(run_reads_fn);
  }
  testing::StartTiming();
  all_read_threads_scheduled_.Notify();

  // Note that destructing the threadpool blocks on completion of all scheduled
  // execution.
  // This is intentional as we want all threads to complete iters iterations.
  // It also means that the timing may be off (work done == iters *
  // num_threads) and includes time scheduling work on the threads.
}

static void BenchmarkReadsAndUpdates(int update_micros, bool do_work, int iters,
                                     int num_threads) {
  BenchmarkState state(update_micros, do_work);
  state.Setup();
  state.RunBenchmarkReadIterations(num_threads, iters);
  state.Teardown();
}

static void BM_Work_NoUpdates_Reads(int iters, int num_threads) {
  // No updates. 0 update_micros signals not to update at all.
  BenchmarkReadsAndUpdates(0, true, iters, num_threads);
}

static void BM_Work_FrequentUpdates_Reads(int iters, int num_threads) {
  // Frequent updates: 1000 micros == 1 millisecond or 1000qps of updates
  BenchmarkReadsAndUpdates(1000, true, iters, num_threads);
}

static void BM_NoWork_NoUpdates_Reads(int iters, int num_threads) {
  // No updates. 0 update_micros signals not to update at all.
  BenchmarkReadsAndUpdates(0, false, iters, num_threads);
}

static void BM_NoWork_FrequentUpdates_Reads(int iters, int num_threads) {
  // Frequent updates: 1000 micros == 1 millisecond or 1000qps of updates
  BenchmarkReadsAndUpdates(1000, false, iters, num_threads);
}

BENCHMARK(BM_Work_NoUpdates_Reads)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(BM_Work_FrequentUpdates_Reads)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(BM_NoWork_NoUpdates_Reads)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(BM_NoWork_FrequentUpdates_Reads)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

}  // namespace serving
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::testing::RunBenchmarks();
  return 0;
}
