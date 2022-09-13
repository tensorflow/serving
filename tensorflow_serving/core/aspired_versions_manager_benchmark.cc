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

// Run with:
// bazel run -c opt --dynamic_mode=off \
// tensorflow_serving/core:aspired_versions_manager_benchmark --
// --benchmarks=.
// For a longer run time and more consistent results, consider a min time
// e.g.: --benchmark_min_time=60.0

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/aspired_version_policy.h"
#include "tensorflow_serving/core/aspired_versions_manager.h"
#include "tensorflow_serving/core/availability_preserving_policy.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/test_util/manager_test_util.h"

namespace tensorflow {
namespace serving {
namespace {

constexpr char kServableName[] = "kServableName";

// Benchmarks for read performance of the AspiredVersionsManager class both with
// and without concurrent updates. These simulate the common expected access
// pattern of AspiredVersionsManager for systems with high read rates and low
// update rates.
//
// This class maintains all state for a benchmark and handles the concurrency
// concerns around the concurrent read and update threads.
//
// Example:
//    BenchmarkState state;
//    state.RunBenchmark(42 /* iters */, 5 /* num_threads */);
class BenchmarkState {
 public:
  BenchmarkState(const int interval_micros, const bool do_work)
      : interval_micros_(interval_micros), do_work_(do_work) {
    AspiredVersionsManager::Options options;
    // Do policy thread won't be run automatically.
    options.manage_state_interval_micros = -1;
    options.aspired_version_policy.reset(new AvailabilityPreservingPolicy());
    TF_CHECK_OK(AspiredVersionsManager::Create(std::move(options), &manager_));
  }

  // Actually perform iters reads on the fast read ptr.
  void RunBenchmark(::testing::benchmark::State& state, int num_threads);

 private:
  void SetUp();
  void TearDown();

  // Runs iters number of reads.
  void RunReads(int iters);

  // Runs continuously after setup and until teardown, if interval_micros was
  // greater than 0.
  void RunUpdate();

  // Starts serving this loader version.
  void StartServing(int64_t loader_version);

  // Gets the latest version of the loader available for serving.
  int64_t GetLatestVersion(bool do_work);

  // To avoid having the benchmark timing include time spent scheduling threads,
  // we use this notification to notify when the read threads should begin.
  // This is notified immediately after the benchmark timing is started.
  Notification all_read_threads_scheduled_;

  // Store the update thread as it is only safe to complete teardown and
  // destruct state after it has exited.
  std::unique_ptr<PeriodicFunction> update_thread_;

  // The AspiredVersionsManager being benchmarked primarily for read
  // performance.
  std::unique_ptr<AspiredVersionsManager> manager_;

  // Interval in microseconds for running the update thread.
  const int interval_micros_;

  // In each iteration, to simulate a more realistic access pattern that does
  // more than content for the mutex.
  bool do_work_;
};

void BenchmarkState::StartServing(const int64_t loader_version) {
  std::unique_ptr<Loader> loader(new SimpleLoader<int64_t>(
      [loader_version](std::unique_ptr<int64_t>* const servable) {
        servable->reset(new int64_t);
        **servable = loader_version;
        return OkStatus();
      },
      SimpleLoader<int64_t>::EstimateNoResources()));
  std::vector<ServableData<std::unique_ptr<Loader>>> versions;
  versions.push_back({{kServableName, loader_version}, std::move(loader)});
  manager_->GetAspiredVersionsCallback()(kServableName, std::move(versions));
  test_util::AspiredVersionsManagerTestAccess(manager_.get())
      .HandlePendingAspiredVersionsRequests();
  // Will load the latest.
  test_util::AspiredVersionsManagerTestAccess(manager_.get())
      .InvokePolicyAndExecuteAction();
  // Will quiesce the previous.
  test_util::AspiredVersionsManagerTestAccess(manager_.get())
      .InvokePolicyAndExecuteAction();
  // Will delete the previous.
  test_util::AspiredVersionsManagerTestAccess(manager_.get())
      .InvokePolicyAndExecuteAction();
  CHECK_EQ(1, manager_->ListAvailableServableIds().size());
}

int64_t BenchmarkState::GetLatestVersion(const bool do_work) {
  ServableHandle<int64_t> handle;
  const Status status = manager_->GetServableHandle(
      ServableRequest::Latest(kServableName), &handle);
  TF_CHECK_OK(status) << status;
  if (do_work) {
    // Let's do some work, so that we are not just measuring contention in the
    // mutex.
    float count = 0;
    for (int i = 1; i < 10000; ++i) {
      count *= i;
    }
    CHECK_GE(count, 0);
  }

  return *handle;
}

void BenchmarkState::RunUpdate() { StartServing(GetLatestVersion(false) + 1); }

void BenchmarkState::SetUp() {
  StartServing(0);
  if (interval_micros_ > 0) {
    PeriodicFunction::Options pf_options;
    pf_options.thread_name_prefix =
        "AspiredVersionsManager_Benchmark_Update_Thread";
    update_thread_.reset(new PeriodicFunction([this] { RunUpdate(); },
                                              interval_micros_, pf_options));
  }
}

void BenchmarkState::TearDown() {
  // Destruct the update thread which blocks until it exits.
  update_thread_.reset();
}

void BenchmarkState::RunReads(int iters) {
  for (int i = 0; i < iters; ++i) {
    // Prevents the compiler from optimizing this away.
    CHECK_GE(GetLatestVersion(do_work_), 0);
  }
}

void BenchmarkState::RunBenchmark(::testing::benchmark::State& state,
                                  int num_threads) {
  SetUp();

  // To be compatible with the Google benchmark framework, the tensorflow new
  // benchmark API requires that each benchmark routine has exactly one.
  // `for (auto s : state)` benchmark loop (in all threads).
  // Therefore we cannot have multiple threads executing the same for-each loop.
  // We need to introduce a new parameter for the fixed number of iteration in
  // each thread.

  // Pick a reasonably large value.
  const int kSubIters = 500;

  // The benchmark timing loop. Timer automatically starts/stops.
  // In each iteration, we spin up a thread-pool and execute kSubIters in each
  // thread.
  for (auto s : state) {
    // Exclude the scheduling setup time.
    state.PauseTiming();
    std::unique_ptr<thread::ThreadPool> pool(new thread::ThreadPool(
        Env::Default(), "RunBenchmarkReadThread", num_threads));
    for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
      std::function<void()> run_reads_fn = [&]() {
        // Wait until all_read_threads_scheduled_ has been notified.
        all_read_threads_scheduled_.WaitForNotification();
        RunReads(kSubIters);
      };
      pool->Schedule(run_reads_fn);
    }
    state.ResumeTiming();
    if (!all_read_threads_scheduled_.HasBeenNotified())
      all_read_threads_scheduled_.Notify();

    // Note that destructing the threadpool blocks on completion of all
    // scheduled execution.  This is intentional as we want all threads to
    // complete iters iterations.  It also means that the timing may be off
    // (work done == iters * num_threads) and includes time scheduling work on
    // the threads.
    pool.reset();
  }

  state.SetItemsProcessed(num_threads * kSubIters * state.iterations());
  TearDown();
}

void BenchmarkReadsAndUpdates(::testing::benchmark::State& state,
                              int num_threads, int interval_micros,
                              bool do_work) {
  BenchmarkState bm_state(interval_micros, do_work);
  bm_state.RunBenchmark(state, num_threads);
}

void BM_Work_NoUpdates_Reads(::testing::benchmark::State& state) {
  const int num_threads = state.range(0);

  // No updates. 0 interval_micros signals not to update at all.
  BenchmarkReadsAndUpdates(state, num_threads, 0, true);
}

void BM_Work_FrequentUpdates_Reads(::testing::benchmark::State& state) {
  const int num_threads = state.range(0);

  // Frequent updates: 1000 micros == 1 millisecond or 1000qps of updates
  BenchmarkReadsAndUpdates(state, num_threads, 1000, true);
}

void BM_NoWork_NoUpdates_Reads(::testing::benchmark::State& state) {
  const int num_threads = state.range(0);

  // No updates. 0 interval_micros signals not to update at all.
  BenchmarkReadsAndUpdates(state, num_threads, 0, false);
}

void BM_NoWork_FrequentUpdates_Reads(::testing::benchmark::State& state) {
  const int num_threads = state.range(0);

  // Frequent updates: 1000 micros == 1 millisecond or 1000qps of updates
  BenchmarkReadsAndUpdates(state, num_threads, 1000, false);
}

// The benchmarking system by default uses cpu time to calculate items per
// second, which would include time spent by all the threads on the cpu.
// Instead of that we use real-time here so that we can see items/s increasing
// with increasing threads, which is easier to understand.
BENCHMARK(BM_Work_NoUpdates_Reads)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(BM_Work_FrequentUpdates_Reads)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(BM_NoWork_NoUpdates_Reads)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(BM_NoWork_FrequentUpdates_Reads)
    ->UseRealTime()
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

void BM_GetServableHandle(::testing::benchmark::State& state) {
  // Number of different servable streams.
  constexpr int kNumServableStreams = 10;
  // Number of versions of a particular servable stream.
  constexpr int kNumServableVersions = 2;

  static AspiredVersionsManager* const manager = []() {
    AspiredVersionsManager::Options options;
    // Do policy thread won't be run automatically.
    options.manage_state_interval_micros = -1;
    options.aspired_version_policy.reset(new AvailabilityPreservingPolicy());
    std::unique_ptr<AspiredVersionsManager> manager;
    TF_CHECK_OK(AspiredVersionsManager::Create(std::move(options), &manager));
    auto aspired_versions_callback = manager->GetAspiredVersionsCallback();
    for (int i = 0; i < kNumServableStreams; ++i) {
      const string servable_name = strings::StrCat(kServableName, i);
      std::vector<ServableData<std::unique_ptr<Loader>>> versions;
      for (int j = 0; j < kNumServableVersions; ++j) {
        std::unique_ptr<Loader> loader(new SimpleLoader<int64_t>(
            [j](std::unique_ptr<int64_t>* const servable) {
              servable->reset(new int64_t);
              **servable = j;
              return OkStatus();
            },
            SimpleLoader<int64_t>::EstimateNoResources()));
        versions.push_back({{servable_name, j}, std::move(loader)});
      }

      aspired_versions_callback(servable_name, std::move(versions));
      test_util::AspiredVersionsManagerTestAccess(manager.get())
          .HandlePendingAspiredVersionsRequests();
      for (int j = 0; j < kNumServableVersions; ++j) {
        test_util::AspiredVersionsManagerTestAccess(manager.get())
            .InvokePolicyAndExecuteAction();
      }
    }
    return manager.release();
  }();
  CHECK_EQ(kNumServableStreams * kNumServableVersions,
           manager->ListAvailableServableIds().size());

  constexpr int kNumRequests = 1024;
  // Ratio of requests which are asking for the latest servable as opposed to a
  // specific version.
  constexpr float kLatestRatio = 0.8;
  static const std::vector<ServableRequest>* requests = []() {
    std::vector<ServableRequest>* requests(new std::vector<ServableRequest>());
    random::PhiloxRandom philox(testing::RandomSeed());
    random::SimplePhilox random(&philox);
    for (int i = 0; i < kNumRequests; ++i) {
      const string name =
          strings::StrCat(kServableName, random.Uniform(kNumServableStreams));
      if (random.RandFloat() > kLatestRatio) {
        const int64_t version = random.Uniform(kNumServableVersions);
        requests->push_back(ServableRequest::Specific(name, version));
      } else {
        requests->push_back(ServableRequest::Latest(name));
      }
    }
    return requests;
  }();

  ServableHandle<int64_t> handle;
  int i = 0;
  for (auto s : state) {
    const Status status =
        manager->GetServableHandle(requests->at(i % kNumRequests), &handle);
    TF_CHECK_OK(status) << status;
    ++i;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_GetServableHandle);

}  // namespace
}  // namespace serving
}  // namespace tensorflow

int main(int argc, char** argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  tensorflow::testing::RunBenchmarks();
  return 0;
}
