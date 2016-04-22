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

#include "tensorflow_serving/core/simple_loader.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using test_util::CreateProto;
using test_util::EqualsProto;

// State reflects the current state of a Caller object.
enum class State {
  kNone,
  kCtor,
  kDoStuff,
  kDtor,
};

// Caller updates a handle to State as it is created, used and destroyed.
class Caller {
 public:
  explicit Caller(State* state) : state_(state) { *state_ = State::kCtor; }
  void DoStuff() { *state_ = State::kDoStuff; }
  ~Caller() { *state_ = State::kDtor; }

 private:
  State* state_;
};

// Move a Loader through its lifetime and ensure the servable is in the state
// we expect.
TEST(SimpleLoader, VerifyServableStates) {
  State state = State::kNone;
  std::unique_ptr<Loader> loader(new SimpleLoader<Caller>(
      [&state](std::unique_ptr<Caller>* caller) {
        caller->reset(new Caller(&state));
        return Status::OK();
      },
      SimpleLoader<Caller>::EstimateNoResources()));
  EXPECT_EQ(State::kNone, state);
  const Status status = loader->Load(ResourceAllocation());
  TF_EXPECT_OK(status);
  EXPECT_EQ(State::kCtor, state);
  AnyPtr servable = loader->servable();
  ASSERT_TRUE(servable.get<Caller>() != nullptr);
  servable.get<Caller>()->DoStuff();
  EXPECT_EQ(State::kDoStuff, state);
  loader->Unload();
  EXPECT_EQ(State::kDtor, state);
  state = State::kNone;
  loader.reset(nullptr);
  EXPECT_EQ(State::kNone, state);
}

TEST(SimpleLoader, ResourceEstimation) {
  const auto want = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'main' "
      "    kind: 'processing' "
      "  } "
      "  quantity: 42 "
      "} ");
  std::unique_ptr<Loader> loader(new SimpleLoader<int>(
      [](std::unique_ptr<int>* servable) {
        servable->reset(new int);
        return Status::OK();
      },
      [&want](ResourceAllocation* estimate) {
        *estimate = want;
        return Status::OK();
      }));
  for (int i = 0; i < 2; ++i) {
    ResourceAllocation got;
    TF_ASSERT_OK(loader->EstimateResources(&got));
    EXPECT_THAT(got, EqualsProto(want));
  }
}

// Verify that the error returned by the Creator is propagates back through
// Load.
TEST(SimpleLoader, LoadError) {
  std::unique_ptr<Loader> loader(new SimpleLoader<Caller>(
      [](std::unique_ptr<Caller>* caller) {
        return errors::InvalidArgument("No way!");
      },
      SimpleLoader<Caller>::EstimateNoResources()));
  const Status status = loader->Load(ResourceAllocation());
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
  EXPECT_EQ("No way!", status.error_message());
}

TEST(SimpleLoaderSourceAdapter, Basic) {
  SimpleLoaderSourceAdapter<string, string> adapter(
      [](const string& data, std::unique_ptr<string>* servable) {
        servable->reset(new string);
        **servable = strings::StrCat(data, "_was_here");
        return Status::OK();
      },
      [](const string& data, ResourceAllocation* output) {
        ResourceAllocation::Entry* entry = output->add_resource_quantities();
        entry->mutable_resource()->set_device(data);
        entry->set_quantity(42);
        return Status::OK();
      });

  const string kServableName = "test_servable_name";
  bool callback_called;
  adapter.SetAspiredVersionsCallback(
      [&](const StringPiece servable_name,
          std::vector<ServableData<std::unique_ptr<Loader>>> versions) {
        callback_called = true;
        EXPECT_EQ(kServableName, servable_name);
        EXPECT_EQ(1, versions.size());
        TF_ASSERT_OK(versions[0].status());
        std::unique_ptr<Loader> loader = versions[0].ConsumeDataOrDie();
        ResourceAllocation estimate_given;
        TF_ASSERT_OK(loader->EstimateResources(&estimate_given));
        EXPECT_THAT(estimate_given, EqualsProto(CreateProto<ResourceAllocation>(
                                        "resource_quantities { "
                                        "  resource { "
                                        "    device: 'test_data' "
                                        "  } "
                                        "  quantity: 42 "
                                        "} ")));
        TF_ASSERT_OK(loader->Load(ResourceAllocation()));
        AnyPtr servable = loader->servable();
        ASSERT_TRUE(servable.get<string>() != nullptr);
        EXPECT_EQ("test_data_was_here", *servable.get<string>());
      });
  adapter.SetAspiredVersions(
      kServableName, {ServableData<string>({kServableName, 0}, "test_data")});
  EXPECT_TRUE(callback_called);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
