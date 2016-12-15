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

#include "tensorflow_serving/util/class_registration.h"

#include <map>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow_serving/util/class_registration_test.pb.h"

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace tensorflow {
namespace serving {
namespace {

// A base class and associated registry.
class MyBaseClass {
 public:
  virtual ~MyBaseClass() = default;
  virtual string class_name() const = 0;
  virtual string config_data() const = 0;
};
DEFINE_CLASS_REGISTRY(MyBaseClassRegistry, MyBaseClass);
#define REGISTER_MY_BASE_CLASS(SubClassCreator, ConfigProto)        \
  REGISTER_CLASS(MyBaseClassRegistry, MyBaseClass, SubClassCreator, \
                 ConfigProto);

// A subclass of MyBaseClass that should be instantiated via Config1.
class SubClass1 : public MyBaseClass {
 public:
  static Status Create(const Config1& config,
                       std::unique_ptr<MyBaseClass>* result) {
    if (config.GetDescriptor()->full_name() != "tensorflow.serving.Config1") {
      return errors::InvalidArgument("Wrong type of config proto: ",
                                     config.GetDescriptor()->full_name());
    }
    auto* raw_result = new SubClass1();
    result->reset(raw_result);
    raw_result->config_ = config;
    return Status::OK();
  }

  string class_name() const override { return "SubClass1"; }

  string config_data() const override { return config_.string_field(); }

 private:
  Config1 config_;
};
REGISTER_MY_BASE_CLASS(SubClass1, Config1);

// A subclass of MyBaseClass that should be instantiated via Config2.
class SubClass2 : public MyBaseClass {
 public:
  static Status Create(const Config2& config,
                       std::unique_ptr<MyBaseClass>* result) {
    if (config.GetDescriptor()->full_name() != "tensorflow.serving.Config2") {
      return errors::InvalidArgument("Wrong type of config proto: ",
                                     config.GetDescriptor()->full_name());
    }
    auto* raw_result = new SubClass2();
    result->reset(raw_result);
    raw_result->config_ = config;
    return Status::OK();
  }

  string class_name() const override { return "SubClass2"; }

  string config_data() const override { return config_.string_field(); }

 private:
  Config2 config_;
};

// A creator of SubClass2 objects.
class SubClass2Creator {
 public:
  static Status Create(const Config2& config,
                       std::unique_ptr<MyBaseClass>* result) {
    return SubClass2::Create(config, result);
  }
};
REGISTER_MY_BASE_CLASS(SubClass2Creator, Config2);

TEST(ClassRegistrationTest, InstantiateFromRawConfig) {
  std::unique_ptr<MyBaseClass> loaded_subclass;

  Config1 config1;
  config1.set_string_field("foo");
  TF_ASSERT_OK(MyBaseClassRegistry::Create(config1, &loaded_subclass));
  EXPECT_EQ("SubClass1", loaded_subclass->class_name());
  EXPECT_EQ("foo", loaded_subclass->config_data());

  Config2 config2;
  config2.set_string_field("bar");
  TF_ASSERT_OK(MyBaseClassRegistry::Create(config2, &loaded_subclass));
  EXPECT_EQ("SubClass2", loaded_subclass->class_name());
  EXPECT_EQ("bar", loaded_subclass->config_data());
}

TEST(ClassRegistrationTest, InstantiateFromAny) {
  std::unique_ptr<MyBaseClass> loaded_subclass;

  Config1 config1;
  config1.set_string_field("foo");
  google::protobuf::Any any_config1;
  any_config1.PackFrom(config1);
  TF_ASSERT_OK(
      MyBaseClassRegistry::CreateFromAny(any_config1, &loaded_subclass));
  EXPECT_EQ("SubClass1", loaded_subclass->class_name());
  EXPECT_EQ("foo", loaded_subclass->config_data());

  Config2 config2;
  config2.set_string_field("bar");
  google::protobuf::Any any_config2;
  any_config2.PackFrom(config2);
  TF_ASSERT_OK(
      MyBaseClassRegistry::CreateFromAny(any_config2, &loaded_subclass));
  EXPECT_EQ("SubClass2", loaded_subclass->class_name());
  EXPECT_EQ("bar", loaded_subclass->config_data());
}

// A second registry for MyBaseClass, with a different name.
DEFINE_CLASS_REGISTRY(AlternateMyBaseClassRegistry, MyBaseClass);
#define REGISTER_MY_BASE_CLASS_USING_ALTERNATE_REGISTRY(SubClassCreator,     \
                                                        ConfigProto)         \
  REGISTER_CLASS(AlternateMyBaseClassRegistry, MyBaseClass, SubClassCreator, \
                 ConfigProto);

// A subclass of MyBaseClass that should be instantiated via Config1, and is
// registered in the alternate registry.
class AlternateSubClass : public MyBaseClass {
 public:
  static Status Create(const Config1& config,
                       std::unique_ptr<MyBaseClass>* result) {
    if (config.GetDescriptor()->full_name() != "tensorflow.serving.Config1") {
      return errors::InvalidArgument("Wrong type of config proto: ",
                                     config.GetDescriptor()->full_name());
    }
    auto* raw_result = new AlternateSubClass();
    result->reset(raw_result);
    raw_result->config_ = config;
    return Status::OK();
  }

  string class_name() const override { return "AlternateSubClass"; }

  string config_data() const override { return config_.string_field(); }

 private:
  Config1 config_;
};
REGISTER_MY_BASE_CLASS_USING_ALTERNATE_REGISTRY(AlternateSubClass, Config1);

TEST(ClassRegistrationTest, MultipleRegistriesForSameBaseClass) {
  std::unique_ptr<MyBaseClass> loaded_subclass;

  Config1 config;
  TF_ASSERT_OK(MyBaseClassRegistry::Create(config, &loaded_subclass));
  EXPECT_EQ("SubClass1", loaded_subclass->class_name());

  TF_ASSERT_OK(AlternateMyBaseClassRegistry::Create(config, &loaded_subclass));
  EXPECT_EQ("AlternateSubClass", loaded_subclass->class_name());
}

// A base class whose subclasses' Create() methods take additional parameters,
// and associated registry.
class MyParameterizedBaseClass {
 public:
  virtual ~MyParameterizedBaseClass() = default;
  virtual string class_name() const = 0;
  virtual string config_data() const = 0;
  virtual int param1_data() const = 0;
  virtual string param2_data() const = 0;
  virtual const std::map<string, int>& param3_data() const = 0;
};
DEFINE_CLASS_REGISTRY(MyParameterizedBaseClassRegistry,
                      MyParameterizedBaseClass, int, const string&,
                      const std::map<string COMMA int>&);
#define REGISTER_MY_PARAMETERIZED_BASE_CLASS(SubClassCreator, ConfigProto)   \
  REGISTER_CLASS(MyParameterizedBaseClassRegistry, MyParameterizedBaseClass, \
                 SubClassCreator, ConfigProto, int, const string&,           \
                 const std::map<string COMMA int>&);

// A subclass of MyParameterizedBaseClass that should be instantiated via
// Config1.
class ParameterizedSubClass1 : public MyParameterizedBaseClass {
 public:
  static Status Create(const Config1& config, int param1, const string& param2,
                       const std::map<string, int>& param3,
                       std::unique_ptr<MyParameterizedBaseClass>* result) {
    if (config.GetDescriptor()->full_name() != "tensorflow.serving.Config1") {
      return errors::InvalidArgument("Wrong type of config proto: ",
                                     config.GetDescriptor()->full_name());
    }
    auto* raw_result = new ParameterizedSubClass1();
    result->reset(raw_result);
    raw_result->config_ = config;
    raw_result->param1_ = param1;
    raw_result->param2_ = param2;
    raw_result->param3_ = param3;
    return Status::OK();
  }

  string class_name() const override { return "ParameterizedSubClass1"; }

  string config_data() const override { return config_.string_field(); }

  int param1_data() const override { return param1_; }

  string param2_data() const override { return param2_; }

  const std::map<string, int>& param3_data() const override { return param3_; }

 private:
  Config1 config_;
  int param1_;
  string param2_;
  std::map<string, int> param3_;
};
REGISTER_MY_PARAMETERIZED_BASE_CLASS(ParameterizedSubClass1, Config1);

TEST(ClassRegistrationTest, InstantiateParameterizedFromRawConfig) {
  std::unique_ptr<MyParameterizedBaseClass> loaded_subclass;

  Config1 config1;
  config1.set_string_field("foo");
  TF_ASSERT_OK(MyParameterizedBaseClassRegistry::Create(
      config1, 42, "bar", {{"floop", 1}, {"mrop", 2}}, &loaded_subclass));
  EXPECT_EQ("ParameterizedSubClass1", loaded_subclass->class_name());
  EXPECT_EQ("foo", loaded_subclass->config_data());
  EXPECT_EQ(42, loaded_subclass->param1_data());
  EXPECT_EQ("bar", loaded_subclass->param2_data());
  EXPECT_THAT(loaded_subclass->param3_data(),
              UnorderedElementsAre(Pair("floop", 1), Pair("mrop", 2)));
}

TEST(ClassRegistrationTest, InstantiateParameterizedFromAny) {
  std::unique_ptr<MyParameterizedBaseClass> loaded_subclass;

  Config1 config1;
  config1.set_string_field("foo");
  google::protobuf::Any any_config1;
  any_config1.PackFrom(config1);
  TF_ASSERT_OK(MyParameterizedBaseClassRegistry::CreateFromAny(
      any_config1, 42, "bar", {{"floop", 1}, {"mrop", 2}}, &loaded_subclass));
  EXPECT_EQ("ParameterizedSubClass1", loaded_subclass->class_name());
  EXPECT_EQ("foo", loaded_subclass->config_data());
  EXPECT_EQ(42, loaded_subclass->param1_data());
  EXPECT_EQ("bar", loaded_subclass->param2_data());
  EXPECT_THAT(loaded_subclass->param3_data(),
              UnorderedElementsAre(Pair("floop", 1), Pair("mrop", 2)));
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
