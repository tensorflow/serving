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

// A gRPC server that serves inception model exported by inception_export.py.
// Given each request with an image as JPEG encoded byte stream, the server
// responds with kNumTopClasses integer values as indexes of top k matched
// categories and kNumTopClasses float values as the corresponding
// probabilities.

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <iostream>

#include "grpc++/completion_queue.h"
#include "grpc++/security/server_credentials.h"
#include "grpc++/server.h"
#include "grpc++/server_builder.h"
#include "grpc++/server_context.h"
#include "grpc++/support/async_unary_call.h"
#include "grpc++/support/status.h"
#include "grpc++/support/status_code_enum.h"
#include "grpc/grpc.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/batching/basic_batch_scheduler.h"
#include "tensorflow_serving/batching/batch_scheduler.h"
#include "tensorflow_serving/batching/batch_scheduler_retrier.h"
#include "tensorflow_serving/core/manager.h"
#include "tensorflow_serving/core/servable_handle.h"
#include "tensorflow_serving/core/servable_id.h"
#include "tensorflow_serving/example/inception_inference.grpc.pb.h"
#include "tensorflow_serving/example/inception_inference.pb.h"
#include "tensorflow_serving/servables/tensorflow/simple_servers.h"
#include "tensorflow_serving/session_bundle/manifest.pb.h"
#include "tensorflow_serving/session_bundle/session_bundle.h"
#include "tensorflow_serving/session_bundle/signature.h"

using grpc::InsecureServerCredentials;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using grpc::StatusCode;
using tensorflow::serving::InceptionRequest;
using tensorflow::serving::InceptionResponse;
using tensorflow::serving::InceptionService;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::serving::ClassificationSignature;

const int kNumTopClasses = 5;

namespace {
class InceptionServiceImpl;

// Class encompassing the state and logic needed to serve a request.
class CallData {
 public:
  CallData(InceptionServiceImpl* service_impl,
           InceptionService::AsyncService* service,
           ServerCompletionQueue* cq);

  void Proceed();

  void Finish(Status status);

  const InceptionRequest& request() { return request_; }

  InceptionResponse* mutable_response() { return &response_; }

 private:
  // Service implementation.
  InceptionServiceImpl* service_impl_;

  // The means of communication with the gRPC runtime for an asynchronous
  // server.
  InceptionService::AsyncService* service_;
  // The producer-consumer queue where for asynchronous server notifications.
  ServerCompletionQueue* cq_;
  // Context for the rpc, allowing to tweak aspects of it such as the use
  // of compression, authentication, as well as to send metadata back to the
  // client.
  ServerContext ctx_;

  // What we get from the client.
  InceptionRequest request_;
  // What we send back to the client.
  InceptionResponse response_;

  // The means to get back to the client.
  ServerAsyncResponseWriter<InceptionResponse> responder_;

  // Let's implement a tiny state machine with the following states.
  enum CallStatus { CREATE, PROCESS, FINISH };
  CallStatus status_;  // The current serving state.
};

// A Task holds all of the information for a single inference request.
struct Task : public tensorflow::serving::BatchTask {
  ~Task() override = default;
  size_t size() const override { return 1; }

  Task(CallData* calldata_arg)
      : calldata(calldata_arg) {}

  CallData* calldata;
};

class InceptionServiceImpl final {
 public:
  InceptionServiceImpl(const string& servable_name,
                       std::unique_ptr<tensorflow::serving::Manager> manager);

  void Classify(CallData* call_data);

  // Produces classifications for a batch of requests and associated responses.
  void DoClassifyInBatch(
      std::unique_ptr<tensorflow::serving::Batch<Task>> batch);

  // Name of the servable to use for inference.
  const string servable_name_;
  // Manager in charge of loading and unloading servables.
  std::unique_ptr<tensorflow::serving::Manager> manager_;
  // A scheduler for batching multiple request calls into single calls to
  // Session->Run().
  std::unique_ptr<tensorflow::serving::BatchScheduler<Task>> batch_scheduler_;
};

// Take in the "service" instance (in this case representing an asynchronous
// server) and the completion queue "cq" used for asynchronous communication
// with the gRPC runtime.
CallData::CallData(InceptionServiceImpl* service_impl,
                   InceptionService::AsyncService* service,
                   ServerCompletionQueue* cq)
    : service_impl_(service_impl),
      service_(service), cq_(cq), responder_(&ctx_), status_(CREATE) {
  // Invoke the serving logic right away.
  Proceed();
}

void CallData::Proceed() {
  if (status_ == CREATE) {
    // As part of the initial CREATE state, we *request* that the system
    // start processing Classify requests. In this request, "this" acts are
    // the tag uniquely identifying the request (so that different CallData
    // instances can serve different requests concurrently), in this case
    // the memory address of this CallData instance.
    service_->RequestClassify(&ctx_, &request_, &responder_, cq_, cq_, this);
    // Make this instance progress to the PROCESS state.
    status_ = PROCESS;
  } else if (status_ == PROCESS) {
    // Spawn a new CallData instance to serve new clients while we process
    // the one for this CallData. The instance will deallocate itself as
    // part of its FINISH state.
    new CallData(service_impl_, service_, cq_);
    // Start processing.
    service_impl_->Classify(this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    // Once in the FINISH state, deallocate ourselves (CallData).
    delete this;
  }
}

void CallData::Finish(Status status) {
  status_ = FINISH;
  responder_.Finish(response_, status, this);
}

InceptionServiceImpl::InceptionServiceImpl(
    const string& servable_name,
    std::unique_ptr<tensorflow::serving::Manager> manager)
    : servable_name_(servable_name), manager_(std::move(manager)) {
  // Setup a batcher used to combine multiple requests (tasks) into a single
  // graph run for efficiency.
  // The batcher queues tasks until,
  //  (a) the next task would cause the batch to exceed the size target;
  //  (b) waiting for more tasks to be added would exceed the timeout.
  // at which point it processes the entire batch.
  //
  // Use the default batch-size, timeout and thread options.  In general
  // the numbers are extremely performance critical and should be tuned based
  // specific graph structure and usage.
  tensorflow::serving::BasicBatchScheduler<Task>::Options scheduler_options;
  scheduler_options.thread_pool_name = "inception_service_batch_threads";
  // TODO(27776734): Current exported model supports only batch_size=1
  // See inception_export.py for details.
  scheduler_options.max_batch_size = 1;
  tensorflow::serving::BatchSchedulerRetrier<Task>::Options retry_options;
  // Retain the default retry options.
  TF_CHECK_OK(tensorflow::serving::CreateRetryingBasicBatchScheduler<Task>(
      scheduler_options, retry_options,
      [this](std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
        this->DoClassifyInBatch(std::move(batch));
      },
      &batch_scheduler_));
}

// Creates a gRPC Status from a TensorFlow Status.
Status ToGRPCStatus(const tensorflow::Status& status) {
  return Status(static_cast<grpc::StatusCode>(status.code()),
                status.error_message());
}

void InceptionServiceImpl::Classify(CallData* calldata) {
  // Create and submit a task to the batch scheduler.
  std::unique_ptr<Task> task(new Task(calldata));
  tensorflow::Status status = batch_scheduler_->Schedule(&task);

  if (!status.ok()) {
    calldata->Finish(ToGRPCStatus(status));
    return;
  }
}

// Produces classifications for a batch of requests and associated responses.
void InceptionServiceImpl::DoClassifyInBatch(
    std::unique_ptr<tensorflow::serving::Batch<Task>> batch) {
  batch->WaitUntilClosed();
  if (batch->empty()) {
    return;
  }
  const int batch_size = batch->num_tasks();

  // Replies to each task with the given error status.
  auto complete_with_error = [&batch](StatusCode code, const string& msg) {
    Status status(code, msg);
    for (int i = 0; i < batch->num_tasks(); i++) {
      Task* task = batch->mutable_task(i);
      task->calldata->Finish(status);
    }
  };

  // Get a handle to the SessionBundle.  The handle ensures the Manager does
  // not reload this while it is in use.
  auto handle_request =
      tensorflow::serving::ServableRequest::Latest(servable_name_);
  tensorflow::serving::ServableHandle<tensorflow::serving::SessionBundle>
      bundle;
  const tensorflow::Status lookup_status =
      manager_->GetServableHandle(handle_request, &bundle);
  if (!lookup_status.ok()) {
    complete_with_error(StatusCode::INTERNAL,
                        lookup_status.error_message());
    return;
  }

  // Get the default signature of the graph.  Expected to be a
  // classification signature.
  tensorflow::serving::ClassificationSignature signature;
  const tensorflow::Status signature_status =
      GetClassificationSignature(bundle->meta_graph_def, &signature);
  if (!signature_status.ok()) {
    complete_with_error(StatusCode::INTERNAL,
                        signature_status.error_message());
    return;
  }

  // Transform protobuf input to inference input tensor.
  tensorflow::Tensor batched_input(tensorflow::DT_STRING, {batch_size});
  for (int i = 0; i < batch_size; ++i) {
    batched_input.vec<string>()(i) =
        batch->mutable_task(i)->calldata->request().jpeg_encoded();
  }

  // Run classification.
  tensorflow::Tensor batched_classes;
  tensorflow::Tensor batched_scores;
  const tensorflow::Status run_status =
      RunClassification(signature, batched_input, bundle->session.get(),
                        &batched_classes, &batched_scores);
  if (!run_status.ok()) {
    complete_with_error(StatusCode::INTERNAL, run_status.error_message());
    return;
  }

  // Transform inference output tensor to protobuf output.
  for (int i = 0; i < batch_size; ++i) {
    auto calldata = batch->mutable_task(i)->calldata;
    auto classes = calldata->mutable_response()->mutable_classes();
    auto scores = calldata->mutable_response()->mutable_scores();
    for (int j = 0; j < kNumTopClasses; j++) {
      *classes->Add() = batched_classes.matrix<string>()(i, j);
      scores->Add(batched_scores.matrix<float>()(i, j));
    }
    calldata->Finish(Status::OK);
  }
}

void HandleRpcs(InceptionServiceImpl* service_impl,
                InceptionService::AsyncService* service,
                ServerCompletionQueue* cq) {
  // Spawn a new CallData instance to serve new clients.
  new CallData(service_impl, service, cq);
  void* tag;  // uniquely identifies a request.
  bool ok;
  while (true) {
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    cq->Next(&tag, &ok);
    GPR_ASSERT(ok);
    static_cast<CallData*>(tag)->Proceed();
  }
}

// Runs InceptionService server until shutdown.
void RunServer(const int port, const string& servable_name,
               std::unique_ptr<tensorflow::serving::Manager> manager) {
  // "0.0.0.0" is the way to listen on localhost in gRPC.
  const string server_address = "0.0.0.0:" + std::to_string(port);

  InceptionService::AsyncService service;
  ServerBuilder builder;
  std::shared_ptr<grpc::ServerCredentials> creds = InsecureServerCredentials();
  builder.AddListeningPort(server_address, creds);
  builder.RegisterService(&service);
  std::unique_ptr<ServerCompletionQueue> cq = builder.AddCompletionQueue();
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG(INFO) << "Running...";

  InceptionServiceImpl service_impl(servable_name, std::move(manager));
  HandleRpcs(&service_impl, &service, cq.get());
}

}  // namespace

int main(int argc, char** argv) {
  // Parse command-line options.
  tensorflow::int32 port = 0;
  const bool parse_result =
      tensorflow::ParseFlags(&argc, argv, {tensorflow::Flag("port", &port)});
  if (!parse_result) {
    LOG(FATAL) << "Error parsing command line flags.";
  }

  if (argc != 2) {
    LOG(ERROR) << "Usage: inception_inference --port=9000 /path/to/exports";
    return -1;
  }
  const string export_base_path(argv[1]);
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  std::unique_ptr<tensorflow::serving::Manager> manager;
  tensorflow::Status status = tensorflow::serving::simple_servers::
      CreateSingleTFModelManagerFromBasePath(export_base_path, &manager);

  TF_CHECK_OK(status) << "Error creating manager";

  // Wait until at least one model is loaded.
  std::vector<tensorflow::serving::ServableId> ready_ids;
  // TODO(b/25545573): Create a more streamlined startup mechanism than polling.
  do {
    LOG(INFO) << "Waiting for models to be loaded...";
    tensorflow::Env::Default()->SleepForMicroseconds(1 * 1000 * 1000 /*1 sec*/);
    ready_ids = manager->ListAvailableServableIds();
  } while (ready_ids.empty());

  // Run the service.
  RunServer(port, ready_ids[0].name, std::move(manager));

  return 0;
}
