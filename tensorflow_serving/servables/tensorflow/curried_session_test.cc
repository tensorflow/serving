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

#include "tensorflow_serving/servables/tensorflow/curried_session.h"

#include <gmock/gmock.h>
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/threadpool_options.h"
#include "tensorflow_serving/core/test_util/mock_session.h"
#include "tensorflow_serving/test_util/test_util.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::Ref;
using ::testing::Return;

using test_util::EqualsProto;

MATCHER_P(EqualsTensor, value, "") {
  return arg.DebugString() == value.DebugString();
}

class MockThreadPool : public thread::ThreadPoolInterface {
 public:
  MOCK_METHOD(void, Schedule, (std::function<void()>), (override));
  MOCK_METHOD(void, Cancel, (), (override));
  MOCK_METHOD(int, NumThreads, (), (const, override));
  MOCK_METHOD(int, CurrentThreadId, (), (const, override));
};

TEST(CurriedSessionTest, ZeroCurriedInputs) {
  const Tensor input = test::AsScalar(0);

  test_util::MockSession* mock = new test_util::MockSession;
  auto curried = std::unique_ptr<Session>(
      new CurriedSession(std::unique_ptr<Session>(mock), {}));

  EXPECT_CALL(*mock, Run(ElementsAre(Pair("input", EqualsTensor(input))),
                         ElementsAre("output"), ElementsAre("target"), _))
      .WillOnce(Return(Status::OK()));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(
      curried->Run({{"input", input}}, {"output"}, {"target"}, &outputs));
}

TEST(CurriedSessionTest, Basic) {
  const Tensor input_a = test::AsScalar(0);
  const Tensor input_b = test::AsScalar(1);
  const Tensor curried_0 = test::AsScalar(2);
  const Tensor curried_1 = test::AsScalar(3);

  test_util::MockSession* mock = new test_util::MockSession;
  auto curried = std::unique_ptr<Session>(
      new CurriedSession(std::unique_ptr<Session>(mock),
                         {{"curried_0", curried_0}, {"curried_1", curried_1}}));

  EXPECT_CALL(*mock,
              Run(ElementsAre(Pair("input_a", EqualsTensor(input_a)),
                              Pair("input_b", EqualsTensor(input_b)),
                              Pair("curried_0", EqualsTensor(curried_0)),
                              Pair("curried_1", EqualsTensor(curried_1))),
                  ElementsAre("output_a", "output_b"),
                  ElementsAre("target_a", "target_b"), _))
      .WillOnce(Return(Status::OK()));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(curried->Run({{"input_a", input_a}, {"input_b", input_b}},
                            {"output_a", "output_b"}, {"target_a", "target_b"},
                            &outputs));
}

TEST(CurriedSessionTest, WithOptions) {
  RunOptions run_options;
  run_options.set_timeout_in_ms(42);

  const Tensor input_a = test::AsScalar(0);
  const Tensor input_b = test::AsScalar(1);
  const Tensor curried_0 = test::AsScalar(2);
  const Tensor curried_1 = test::AsScalar(3);

  test_util::MockSession* mock = new test_util::MockSession;
  auto curried = std::unique_ptr<Session>(
      new CurriedSession(std::unique_ptr<Session>(mock),
                         {{"curried_0", curried_0}, {"curried_1", curried_1}}));

  EXPECT_CALL(*mock,
              Run(EqualsProto(run_options),
                  ElementsAre(Pair("input_a", EqualsTensor(input_a)),
                              Pair("input_b", EqualsTensor(input_b)),
                              Pair("curried_0", EqualsTensor(curried_0)),
                              Pair("curried_1", EqualsTensor(curried_1))),
                  ElementsAre("output_a", "output_b"),
                  ElementsAre("target_a", "target_b"), _, _))
      .WillOnce(Return(Status::OK()));
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  TF_ASSERT_OK(curried->Run(run_options,
                            {{"input_a", input_a}, {"input_b", input_b}},
                            {"output_a", "output_b"}, {"target_a", "target_b"},
                            &outputs, &run_metadata));
}

TEST(CurriedSessionTest, ExplicitInputsMatchCurriedInputs) {
  const Tensor t0 = test::AsScalar(0);
  const Tensor t1 = test::AsScalar(1);
  const Tensor t2 = test::AsScalar(2);

  test_util::MockSession* mock = new test_util::MockSession;
  auto curried = std::unique_ptr<Session>(new CurriedSession(
      std::unique_ptr<Session>(mock), {{"t0", t0}, {"t1", t1}}));

  EXPECT_CALL(*mock, Run(_, _, _, _)).Times(0);
  std::vector<Tensor> outputs;
  const Status status =
      curried->Run({{"t1", t1}, {"t2", t2}}, {"output"}, {"target"}, &outputs);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.ToString(),
      HasSubstr("Explicit Run() input has same name as curried input t1"));
}

TEST(CurriedSessionTest, PropagateError) {
  test_util::MockSession* mock = new test_util::MockSession;
  auto curried = std::unique_ptr<Session>(
      new CurriedSession(std::unique_ptr<Session>(mock), {}));

  EXPECT_CALL(*mock, Run(_, _, _, _))
      .WillOnce(Return(errors::Unknown("Tensor clog")));
  std::vector<Tensor> outputs;
  const Status status = curried->Run({}, {}, {}, &outputs);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("Tensor clog"));
}

TEST(CurriedSessionTest, ThreadPoolOptions) {
  const Tensor input_a = test::AsScalar(0);
  const Tensor input_b = test::AsScalar(1);
  const Tensor curried_0 = test::AsScalar(2);
  const Tensor curried_1 = test::AsScalar(3);

  test_util::MockSession* mock = new test_util::MockSession;
  auto curried = std::unique_ptr<Session>(
      new CurriedSession(std::unique_ptr<Session>(mock),
                         {{"curried_0", curried_0}, {"curried_1", curried_1}}));

  thread::ThreadPoolOptions thread_pool_options;
  MockThreadPool mock_threadpool;
  thread_pool_options.inter_op_threadpool = &mock_threadpool;
  thread_pool_options.intra_op_threadpool = &mock_threadpool;
  EXPECT_CALL(
      *mock,
      Run(_,
          ElementsAre(Pair("input_a", EqualsTensor(input_a)),
                      Pair("input_b", EqualsTensor(input_b)),
                      Pair("curried_0", EqualsTensor(curried_0)),
                      Pair("curried_1", EqualsTensor(curried_1))),
          ElementsAre("output_a", "output_b"),
          ElementsAre("target_a", "target_b"), _, _, Ref(thread_pool_options)))
      .WillOnce(Return(Status::OK()));
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  TF_ASSERT_OK(curried->Run(RunOptions(),
                            {{"input_a", input_a}, {"input_b", input_b}},
                            {"output_a", "output_b"}, {"target_a", "target_b"},
                            &outputs, &run_metadata, thread_pool_options));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
