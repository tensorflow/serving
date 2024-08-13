/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/match.h"
#include "xla/tsl/lib/histogram/histogram.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;

using tensorflow::Env;
using tensorflow::EnvTime;
using tensorflow::Thread;
using tensorflow::protobuf::TextFormat;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;
using tsl::histogram::ThreadSafeHistogram;

ABSL_FLAG(std::string, server_port, "", "Target server (host:port)");
ABSL_FLAG(std::string, request, "", "File containing request proto message");
ABSL_FLAG(std::string, model_name, "", "Model name to override in the request");
ABSL_FLAG(int, model_version, -1, "Model version to override in the request");
ABSL_FLAG(int, num_requests, 1, "Total number of requests to send.");
ABSL_FLAG(int, qps, 1, "Rate for sending requests.");
ABSL_FLAG(int, rpc_deadline_ms, 1000, "RPC request deadline in milliseconds.");
ABSL_FLAG(bool, print_rpc_errors, false, "Print RPC errors.");

bool ReadProtoFromFile(const std::string& filename, PredictRequest* req) {
  auto in = std::ifstream(filename);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  return absl::EndsWith(filename, ".pbtxt")
             ? TextFormat::ParseFromString(ss.str(), req)
             : req->ParseFromString(ss.str());
}

class ServingClient {
 public:
  ServingClient(const std::string& server_port)
      : stub_(PredictionService::NewStub(grpc::CreateChannel(
            server_port, grpc::InsecureChannelCredentials()))),
        done_count_(0),
        success_count_(0),
        error_count_(0),
        latency_histogram_(new ThreadSafeHistogram()),
        error_histogram_(new ThreadSafeHistogram(
            // Range from grpc::StatusCode enum.
            {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})) {
    thread_.reset(Env::Default()->StartThread({}, "reaprpcs",
                                              [this]() { ReapCompletions(); }));
  }

  void IssuePredict(const PredictRequest& req) {
    auto state = new RpcState();
    state->context.set_deadline(
        std::chrono::system_clock::now() +
        std::chrono::milliseconds(absl::GetFlag(FLAGS_rpc_deadline_ms)));
    state->start_time = EnvTime::NowMicros();
    std::unique_ptr<ClientAsyncResponseReader<PredictResponse>> rpc(
        stub_->AsyncPredict(&state->context, req, &cq_));
    rpc->Finish(&state->resp, &state->status, (void*)state);
  }

  void ReapCompletions() {
    void* sp;
    bool ok = false;
    while (cq_.Next(&sp, &ok)) {
      done_count_++;
      std::unique_ptr<RpcState> state((RpcState*)sp);
      if (state->status.ok()) {
        success_count_++;
        latency_histogram_->Add(EnvTime::NowMicros() - state->start_time);
      } else {
        error_count_++;
        error_histogram_->Add(state->status.error_code());
        if (absl::GetFlag(FLAGS_print_rpc_errors)) {
          std::cerr << "ERROR: RPC failed code: " << state->status.error_code()
                    << " msg: " << state->status.error_message() << std::endl;
        }
      }
    }
  }

  void WaitForCompletion(int total_rpcs) {
    while (done_count_ < total_rpcs) {
      Env::Default()->SleepForMicroseconds(1000);
    }
    cq_.Shutdown();
    thread_.reset();
  }

  void DumpStats() {
    if (success_count_) {
      std::cout << "Request stats (successful)" << std::endl;
      std::cout << latency_histogram_->ToString() << std::endl;
    }
    if (error_count_) {
      std::cout << "Request stats (errors)" << std::endl;
      std::cout << error_histogram_->ToString() << std::endl;
    }
  }

 private:
  struct RpcState {
    uint64_t start_time;
    ClientContext context;
    PredictResponse resp;
    Status status;
  };
  std::unique_ptr<PredictionService::Stub> stub_;
  CompletionQueue cq_;
  std::unique_ptr<Thread> thread_;
  std::atomic<int> done_count_;
  std::atomic<int> success_count_;
  std::atomic<int> error_count_;
  std::unique_ptr<ThreadSafeHistogram> latency_histogram_;
  std::unique_ptr<ThreadSafeHistogram> error_histogram_;
};

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  if (absl::GetFlag(FLAGS_server_port).empty() ||
      absl::GetFlag(FLAGS_request).empty()) {
    std::cerr << "ERROR: --server_port and --request flags are required."
              << std::endl;
    return 1;
  }

  PredictRequest req;
  if (!ReadProtoFromFile(absl::GetFlag(FLAGS_request), &req)) {
    std::cerr << "ERROR: Failed to parse protobuf from file: "
              << absl::GetFlag(FLAGS_request) << std::endl;
    return 1;
  }
  if (!absl::GetFlag(FLAGS_model_name).empty()) {
    req.mutable_model_spec()->set_name(absl::GetFlag(FLAGS_model_name));
  }
  if (absl::GetFlag(FLAGS_model_version) >= 0) {
    req.mutable_model_spec()->mutable_version()->set_value(
        absl::GetFlag(FLAGS_model_version));
  }

  ServingClient client(absl::GetFlag(FLAGS_server_port));
  std::cout << "Sending " << absl::GetFlag(FLAGS_num_requests)
            << " requests to " << absl::GetFlag(FLAGS_server_port) << " at "
            << absl::GetFlag(FLAGS_qps) << " requests/sec." << std::endl;
  for (int i = 0; i < absl::GetFlag(FLAGS_num_requests); i++) {
    client.IssuePredict(req);
    Env::Default()->SleepForMicroseconds(1000000 / absl::GetFlag(FLAGS_qps));
  }

  std::cout << "Waiting for " << absl::GetFlag(FLAGS_num_requests)
            << " requests to complete..." << std::endl;
  client.WaitForCompletion(absl::GetFlag(FLAGS_num_requests));
  client.DumpStats();
  return 0;
}
