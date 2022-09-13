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
#include "absl/memory/memory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/servable_id.h"
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

Loader::Metadata CreateMetadata() { return {ServableId{"name", 42}}; }

class LoaderCreatorWithoutMetadata {
 public:
  template <typename ServableType, typename... Args>
  static std::unique_ptr<Loader> CreateSimpleLoader(
      typename SimpleLoader<ServableType>::Creator creator, Args... args) {
    return absl::make_unique<SimpleLoader<ServableType>>(creator, args...);
  }

  static Status Load(Loader* loader) { return loader->Load(); }
};

class LoaderCreatorWithMetadata {
 public:
  template <typename ServableType, typename... Args>
  static std::unique_ptr<Loader> CreateSimpleLoader(
      typename SimpleLoader<ServableType>::Creator creator, Args... args) {
    return absl::make_unique<SimpleLoader<ServableType>>(
        [creator](const Loader::Metadata& metadata,
                  std::unique_ptr<ServableType>* servable) {
          const auto& expected_metadata = CreateMetadata();
          EXPECT_EQ(expected_metadata.servable_id, metadata.servable_id);
          return creator(servable);
        },
        args...);
  }

  static Status Load(Loader* loader) {
    return loader->LoadWithMetadata(CreateMetadata());
  }
};

template <typename T>
class SimpleLoaderTest : public ::testing::Test {};
using LoaderCreatorTypes =
    ::testing::Types<LoaderCreatorWithoutMetadata, LoaderCreatorWithMetadata>;
TYPED_TEST_SUITE(SimpleLoaderTest, LoaderCreatorTypes);

// Move a Loader through its lifetime and ensure the servable is in the state
// we expect.
TYPED_TEST(SimpleLoaderTest, VerifyServableStates) {
  State state = State::kNone;
  auto loader = TypeParam::template CreateSimpleLoader<Caller>(
      [&state](std::unique_ptr<Caller>* caller) {
        caller->reset(new Caller(&state));
        return OkStatus();
      },
      SimpleLoader<Caller>::EstimateNoResources());
  EXPECT_EQ(State::kNone, state);
  const Status status = TypeParam::Load(loader.get());
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

TYPED_TEST(SimpleLoaderTest, ResourceEstimation) {
  const auto want = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'main' "
      "    kind: 'processing' "
      "  } "
      "  quantity: 42 "
      "} ");
  auto loader = TypeParam::template CreateSimpleLoader<int>(
      [](std::unique_ptr<int>* servable) {
        servable->reset(new int);
        return OkStatus();
      },
      [&want](ResourceAllocation* estimate) {
        *estimate = want;
        return OkStatus();
      });

  {
    ResourceAllocation got;
    TF_ASSERT_OK(loader->EstimateResources(&got));
    EXPECT_THAT(got, EqualsProto(want));
  }

  // The estimate should remain the same after load.
  TF_ASSERT_OK(TypeParam::Load(loader.get()));
  {
    ResourceAllocation got;
    TF_ASSERT_OK(loader->EstimateResources(&got));
    EXPECT_THAT(got, EqualsProto(want));
  }
}

TYPED_TEST(SimpleLoaderTest, ResourceEstimationWithPostLoadRelease) {
  const auto pre_load_resources = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'main' "
      "    kind: 'processing' "
      "  } "
      "  quantity: 42 "
      "} ");
  const auto post_load_resources = CreateProto<ResourceAllocation>(
      "resource_quantities { "
      "  resource { "
      "    device: 'main' "
      "    kind: 'processing' "
      "  } "
      "  quantity: 17 "
      "} ");
  auto loader = TypeParam::template CreateSimpleLoader<int>(
      [](std::unique_ptr<int>* servable) {
        servable->reset(new int);
        return OkStatus();
      },
      [&pre_load_resources](ResourceAllocation* estimate) {
        *estimate = pre_load_resources;
        return OkStatus();
      },
      absl::make_optional([&post_load_resources](ResourceAllocation* estimate) {
        *estimate = post_load_resources;
        return OkStatus();
      }));

  // Run it twice, to exercise memoization.
  for (int i = 0; i < 2; ++i) {
    ResourceAllocation got;
    TF_ASSERT_OK(loader->EstimateResources(&got));
    EXPECT_THAT(got, EqualsProto(pre_load_resources));
  }

  // The estimate should switch to the post-load one after load.
  TF_ASSERT_OK(TypeParam::Load(loader.get()));
  {
    ResourceAllocation got;
    TF_ASSERT_OK(loader->EstimateResources(&got));
    EXPECT_THAT(got, EqualsProto(post_load_resources));
  }
}

// Verify that the error returned by the Creator is propagates back through
// Load.
TYPED_TEST(SimpleLoaderTest, LoadError) {
  auto loader = TypeParam::template CreateSimpleLoader<Caller>(
      [](std::unique_ptr<Caller>* caller) {
        return errors::InvalidArgument("No way!");
      },
      SimpleLoader<Caller>::EstimateNoResources());
  const Status status = TypeParam::Load(loader.get());
  EXPECT_EQ(error::INVALID_ARGUMENT, status.code());
  EXPECT_EQ("No way!", status.error_message());
}

TEST(SimpleLoaderCompatibilityTest, WithoutMetadata) {
  auto loader_without_metadata = absl::make_unique<SimpleLoader<int>>(
      [](std::unique_ptr<int>* servable) {
        servable->reset(new int);
        return OkStatus();
      },
      SimpleLoader<int>::EstimateNoResources());
  // If the creator without metadata is used, both Load() and LoadWithMetadata()
  // are fine, for compatibility.
  TF_EXPECT_OK(loader_without_metadata->Load());
  TF_EXPECT_OK(loader_without_metadata->LoadWithMetadata(CreateMetadata()));
}

TEST(SimpleLoaderCompatibilityTest, WithMetadata) {
  auto loader_with_metadata = absl::make_unique<SimpleLoader<int>>(
      [](const Loader::Metadata& metadata, std::unique_ptr<int>* servable) {
        const auto& expected_metadata = CreateMetadata();
        EXPECT_EQ(expected_metadata.servable_id, metadata.servable_id);
        servable->reset(new int);
        return OkStatus();
      },
      SimpleLoader<int>::EstimateNoResources());
  // If the creator with metadata is used, we allow only LoadWithMetadata()
  // to be invoked.
  const Status error_status = loader_with_metadata->Load();
  EXPECT_EQ(error::FAILED_PRECONDITION, error_status.code());
  TF_EXPECT_OK(loader_with_metadata->LoadWithMetadata(CreateMetadata()));
}

// A pass-through implementation of SimpleLoaderSourceAdapter, which can be
// instantiated.
template <typename DataType, typename ServableType>
class SimpleLoaderSourceAdapterImpl final
    : public SimpleLoaderSourceAdapter<DataType, ServableType> {
 public:
  SimpleLoaderSourceAdapterImpl(
      typename SimpleLoaderSourceAdapter<DataType, ServableType>::Creator
          creator,
      typename SimpleLoaderSourceAdapter<
          DataType, ServableType>::ResourceEstimator resource_estimator)
      : SimpleLoaderSourceAdapter<DataType, ServableType>(creator,
                                                          resource_estimator) {}
  ~SimpleLoaderSourceAdapterImpl() override { TargetBase<DataType>::Detach(); }
};

TEST(SimpleLoaderSourceAdapterTest, Basic) {
  SimpleLoaderSourceAdapterImpl<string, string> adapter(
      [](const string& data, std::unique_ptr<string>* servable) {
        servable->reset(new string);
        **servable = strings::StrCat(data, "_was_here");
        return OkStatus();
      },
      [](const string& data, ResourceAllocation* output) {
        ResourceAllocation::Entry* entry = output->add_resource_quantities();
        entry->mutable_resource()->set_device(data);
        entry->set_quantity(42);
        return OkStatus();
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
        TF_ASSERT_OK(loader->Load());
        AnyPtr servable = loader->servable();
        ASSERT_TRUE(servable.get<string>() != nullptr);
        EXPECT_EQ("test_data_was_here", *servable.get<string>());
      });
  adapter.SetAspiredVersions(
      kServableName, {ServableData<string>({kServableName, 0}, "test_data")});
  EXPECT_TRUE(callback_called);
}

// This test verifies that deleting a SimpleLoaderSourceAdapter doesn't affect
// the loaders it has emitted. This is a regression test for b/30189916.
TEST(SimpleLoaderSourceAdapterTest, OkayToDeleteAdapter) {
  std::unique_ptr<Loader> loader;
  {
    // Allocate 'adapter' on the heap so ASAN will catch a use-after-free.
    auto adapter = std::unique_ptr<SimpleLoaderSourceAdapter<string, string>>(
        new SimpleLoaderSourceAdapterImpl<string, string>(
            [](const string& data, std::unique_ptr<string>* servable) {
              servable->reset(new string);
              **servable = strings::StrCat(data, "_was_here");
              return OkStatus();
            },
            SimpleLoaderSourceAdapter<string, string>::EstimateNoResources()));

    const string kServableName = "test_servable_name";
    adapter->SetAspiredVersionsCallback(
        [&](const StringPiece servable_name,
            std::vector<ServableData<std::unique_ptr<Loader>>> versions) {
          ASSERT_EQ(1, versions.size());
          TF_ASSERT_OK(versions[0].status());
          loader = versions[0].ConsumeDataOrDie();
        });
    adapter->SetAspiredVersions(
        kServableName, {ServableData<string>({kServableName, 0}, "test_data")});

    // Let 'adapter' fall out of scope and be deleted.
  }

  // We should be able to invoke the resource-estimation and servable-creation
  // callbacks, despite the fact that 'adapter' has been deleted.
  ResourceAllocation estimate_given;
  TF_ASSERT_OK(loader->EstimateResources(&estimate_given));
  TF_ASSERT_OK(loader->Load());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
