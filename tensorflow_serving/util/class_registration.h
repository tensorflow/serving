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

// A way to register subclasses of an abstract base class, and associate a
// config proto message type. Instances can be instantiated from an Any proto
// field that contains a config proto, based on the type and content of the
// config proto.
//
// IMPORTANT: Config protos used in registries must be compiled into the binary.
//
// Each registry has a name. Registry names in each namespace must be distinct.
// A registry is tied to a specific base class and factory signature. It is fine
// to have multiple registries for a given base class, whether having the same
// factory signature or multiple distinct signatures.
//
// Usage:
//
//   // Define a base class.
//   class MyBaseClass {
//     ...
//   };
//
//   // Define a registry that maps from proto message types to subclasses of
//   // MyBaseClass.
//   DEFINE_CLASS_REGISTRY(MyBaseClassRegistry, MyBaseClass);
//
//   // Define a macro used create a specific entry in MyBaseClassRegistry that
//   // maps from ConfigProto to ClassCreator::Create().
//   #define REGISTER_MY_BASE_CLASS(ClassCreator, ConfigProto)
//     REGISTER_CLASS(MyBaseClassRegistry, MyBaseClass, ClassCreator,
//                       ConfigProto);
//
//   // Declare a subclass of MyBaseClass to be created when a OneConfigProto
//   // is passed to the Create*() factory methods.
//   class OneClass : public MyBaseClass {
//    public:
//     static Status Create(const OneConfigProto& config,
//                          std::unique_ptr<BaseClass>* result) {
//       OneClass* raw_result = new OneClass();
//       raw_result->config_ = config;
//       Status status = raw_result->Init();
//       if (status.ok()) {
//         result->reset(raw_result);
//       }
//       return status;
//     }
//
//    private:
//     Status Init() {
//       ... initialize the object based on 'config_'
//     }
//
//     OneConfigProto config_;
//   };
//   REGISTER_MY_BASE_CLASS(OneClass, OneConfigProto);
//
//   // Create an object of type OneClass using the registry to switch on
//   // the type OneConfigProto.
//   OneConfigProto config = ...
//   std::unique_ptr<BaseClass> loaded_subclass;
//   CHECK_OK(MyBaseClassRegistry::Create(config, &loaded_subclass));
//
//   // Same, but starting from an Any message that wraps a OneConfigProto.
//   protobuf::Any any_config = ...  // wraps a OneConfigProto
//   std::unique_ptr<BaseClass> loaded_subclass;
//   CHECK_OK(MyBaseClassRegistry::CreateFromAny(any_config, &loaded_subclass));
//
// Note that the subclass creator need not be the subclass itself. For example:
//
//   class AnotherClass : public MyBaseClass {
//    public:
//     AnotherClass(int a, int b);
//     ...
//   };
//
//   class CreatorForAnotherClass {
//    public:
//     static Status Create(const OneConfigProto& config,
//                          std::unique_ptr<BaseClass>* result) {
//       result->reset(new AnotherClass(config.a(), config.b()));
//       return Status::OK;
//     }
//   };
//
//   REGISTER_MY_BASE_CLASS(CreatorForAnotherClass, OneConfigProto);
//
//
// This mechanism also allows additional parameter passing into the Create()
// factory method.  Consider the following example in which Create() takes an
// int and a string, in addition to the config proto:
//
//   class MyParameterizedBaseClass {
//     ...
//   };
//   DEFINE_CLASS_REGISTRY(MyParameterizedBaseClassRegistry,
//                            MyParameterizedBaseClass, int, const string&
//                            const std::map<int COMMA string>&);
//   #define REGISTER_MY_BASE_CLASS(ClassCreator, ConfigProto)
//     REGISTER_CLASS(MyBaseClassRegistry, MyBaseClass, ClassCreator,
//                       ConfigProto, int, const string&,
//                       const std::map<int COMMA string>&);
//
//   class OneClass : public MyParameterizedBaseClass {
//    public:
//     static Status Create(const OneConfigProto& config,
//                          int param1, const string& param2,
//                          const std::map<int, string>& param3,
//                          std::unique_ptr<BaseClass>* result) {
//       ...
//     }
//     ...
//   };
//
//   OneConfigProto config = ...
//   int int_param = ...
//   string string_param = ...
//   std::map<int, string> map_param = ...
//   std::unique_ptr<BaseClass> loaded_subclass;
//   CHECK_OK(MyParameterizedBaseClassRegistry::Create(config,
//                                                     int_param,
//                                                     string_param,
//                                                     map_param,
//                                                     &loaded_subclass));
//
// The registry name can be anything you choose, and it's fine to have multiple
// registries for the same base class, potentially with different factory
// signatures.
//
// Note that types containing a comma, e.g. std::map<string, int> must use COMMA
// in place of ','.
// TODO(b/24472377): Eliminate the requirement to use COMMA.

#ifndef TENSORFLOW_SERVING_UTIL_CLASS_REGISTRATION_H_
#define TENSORFLOW_SERVING_UTIL_CLASS_REGISTRATION_H_

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>

#include "google/protobuf/any.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow_serving/util/class_registration_util.h"

namespace tensorflow {
namespace serving {
namespace internal {

// The interface for a factory method that takes a protobuf::Message as
// input to construct an object of type BaseClass.
template <typename BaseClass, typename... AdditionalFactoryArgs>
class AbstractClassRegistrationFactory {
 public:
  virtual ~AbstractClassRegistrationFactory() = default;

  // Creates an object using this factory.  Fails if 'config' is not of the
  // expected type.
  virtual Status Create(const protobuf::Message& config,
                        AdditionalFactoryArgs... args,
                        std::unique_ptr<BaseClass>* result) const = 0;
};

// The interface for a factory method that takes a protobuf::Message as
// input to construct an object of type BaseClass.
template <typename BaseClass, typename Class, typename Config,
          typename... AdditionalFactoryArgs>
class ClassRegistrationFactory
    : public AbstractClassRegistrationFactory<BaseClass,
                                              AdditionalFactoryArgs...> {
 public:
  ClassRegistrationFactory()
      : config_descriptor_(Config::default_instance().GetDescriptor()) {}

  // Creates an object using this factory.  Fails if 'config' is not of the
  // expected type.
  Status Create(const protobuf::Message& config, AdditionalFactoryArgs... args,
                std::unique_ptr<BaseClass>* result) const override {
    if (config.GetDescriptor() != config_descriptor_) {
      return errors::InvalidArgument(
          "Supplied config proto of type ", config.GetDescriptor()->full_name(),
          " does not match expected type ", config_descriptor_->full_name());
    }
    return Class::Create(static_cast<const Config&>(config),
                         std::forward<AdditionalFactoryArgs>(args)..., result);
  }

 private:
  const protobuf::Descriptor* const config_descriptor_;

  TF_DISALLOW_COPY_AND_ASSIGN(ClassRegistrationFactory);
};

constexpr char kTypeGoogleApisComPrefix[] = "type.googleapis.com/";

// A static map whose keys are proto message names, and values are
// ClassRegistrationFactories. Includes a Create() factory method that
// performs a lookup in the map and calls Create() on the
// ClassRegistrationFactory it finds.
template <typename RegistryName, typename BaseClass,
          typename... AdditionalFactoryArgs>
class ClassRegistry {
 public:
  using FactoryType =
      AbstractClassRegistrationFactory<BaseClass, AdditionalFactoryArgs...>;

  // Creates an instance of BaseClass based on a config proto.
  static Status Create(const protobuf::Message& config,
                       AdditionalFactoryArgs... args,
                       std::unique_ptr<BaseClass>* result) {
    const string& config_proto_message_type =
        config.GetDescriptor()->full_name();
    auto* factory = LookupFromMap(config_proto_message_type);
    if (factory == nullptr) {
      return errors::InvalidArgument(
          "Couldn't find factory for config proto message type ",
          config_proto_message_type, "\nconfig=", config.DebugString());
    }
    return factory->Create(config, std::forward<AdditionalFactoryArgs>(args)...,
                           result);
  }

  // Creates an instance of BaseClass based on a config proto embedded in an Any
  // message.
  //
  // Requires that the config proto in the Any has a compiled-in descriptor.
  static Status CreateFromAny(const google::protobuf::Any& any_config,
                              AdditionalFactoryArgs... args,
                              std::unique_ptr<BaseClass>* result) {
    // Copy the config to a proto message of the indicated type.
    string full_type_name;
    Status parse_status =
        ParseUrlForAnyType(any_config.type_url(), &full_type_name);
    if (!parse_status.ok()) {
      return parse_status;
    }
    const protobuf::Descriptor* descriptor =
        protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(
            full_type_name);
    if (descriptor == nullptr) {
      return errors::Internal(
          "Unable to find compiled-in proto descriptor of type ",
          full_type_name);
    }
    std::unique_ptr<protobuf::Message> config(
        protobuf::MessageFactory::generated_factory()
            ->GetPrototype(descriptor)
            ->New());
    any_config.UnpackTo(config.get());
    return Create(*config, std::forward<AdditionalFactoryArgs>(args)...,
                  result);
  }

  // Nested class whose instantiation inserts a key/value pair into the factory
  // map.
  class MapInserter {
   public:
    MapInserter(const string& config_proto_message_type, FactoryType* factory) {
      InsertIntoMap(config_proto_message_type, factory);
    }
  };

 private:
  // Inserts a key/value pair into the factory map.
  static void InsertIntoMap(const string& config_proto_message_type,
                            FactoryType* factory) {
    LockableFactoryMap* global_map = GlobalFactoryMap();
    {
      mutex_lock lock(global_map->mu);
      global_map->factory_map.insert({config_proto_message_type, factory});
    }
  }

  // Retrieves a value from the factory map, or returns nullptr if no value was
  // found.
  static FactoryType* LookupFromMap(const string& config_proto_message_type) {
    LockableFactoryMap* global_map = GlobalFactoryMap();
    {
      mutex_lock lock(global_map->mu);
      auto it = global_map->factory_map.find(config_proto_message_type);
      if (it == global_map->factory_map.end()) {
        return nullptr;
      }
      return it->second;
    }
  }

  // A map from proto descriptor names to factories, with a lock.
  struct LockableFactoryMap {
    mutex mu;
    std::unordered_map<string, FactoryType*> factory_map GUARDED_BY(mu);
  };

  // Returns a pointer to the factory map. There is one factory map per set of
  // template parameters of this class.
  static LockableFactoryMap* GlobalFactoryMap() {
    static auto* global_map = []() -> LockableFactoryMap* {
      return new LockableFactoryMap;
    }();
    return global_map;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ClassRegistry);
};

}  // namespace internal

// Used to enable the following macros to work with types containing commas
// (e.g. map<string, int>).
// TODO(b/24472377): Eliminate the requirement to use COMMA, via some fancy
// macro gymnastics.
#define COMMA ,

// Given a base class BaseClass, creates a registry named RegistryName.
#define DEFINE_CLASS_REGISTRY(RegistryName, BaseClass, ...)                   \
  class RegistryName : public ::tensorflow::serving::internal::ClassRegistry< \
                           RegistryName, BaseClass, ##__VA_ARGS__> {};

// Registers a factory that creates subclasses of BaseClass by calling
// ClassCreator::Create().
#define REGISTER_CLASS(RegistryName, BaseClass, ClassCreator, config_proto, \
                       ...)                                                 \
  REGISTER_CLASS_UNIQ_HELPER(__COUNTER__, RegistryName, BaseClass,          \
                             ClassCreator, config_proto, ##__VA_ARGS__)

#define REGISTER_CLASS_UNIQ_HELPER(cnt, RegistryName, BaseClass, ClassCreator, \
                                   config_proto, ...)                          \
  REGISTER_CLASS_UNIQ(cnt, RegistryName, BaseClass, ClassCreator,              \
                      config_proto, ##__VA_ARGS__)

#define REGISTER_CLASS_UNIQ(cnt, RegistryName, BaseClass, ClassCreator,    \
                            config_proto, ...)                             \
  static ::tensorflow::serving::internal::ClassRegistry<                   \
      RegistryName, BaseClass, ##__VA_ARGS__>::MapInserter                 \
      register_class_##cnt(                                                \
          (config_proto::default_instance().GetDescriptor()->full_name()), \
          (new ::tensorflow::serving::internal::ClassRegistrationFactory<  \
              BaseClass, ClassCreator, config_proto, ##__VA_ARGS__>));

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_UTIL_CLASS_REGISTRATION_H_
