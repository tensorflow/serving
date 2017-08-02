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

#include "tensorflow_serving/util/unique_ptr_with_deps.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {
namespace {

// Trait to select overloads and return types for MakeUnique.
template <typename T>
struct MakeUniqueResult {
  using scalar = std::unique_ptr<T>;
};

// MakeUnique<T>(...) is an early implementation of C++14 std::make_unique.
// It is designed to be 100% compatible with std::make_unique so that the
// eventual switchover will be a simple renaming operation.
template <typename T, typename... Args>
typename MakeUniqueResult<T>::scalar MakeUnique(Args&&... args) {  // NOLINT
  return std::unique_ptr<T>(
      new T(std::forward<Args>(args)...));  // NOLINT(build/c++11)
}

class Dep {
 public:
  Dep(const string& name, std::vector<string>* log) : name_(name), log_(log) {}
  ~Dep() { log_->push_back("Deleting " + name_); }
  void f() { log_->push_back(name_ + ".f() called"); }

 private:
  string name_;
  std::vector<string>* log_;
};

// Create another class just to show that dependencies can have different type
// than main object.
class Obj {
 public:
  Obj(const string& name, std::vector<string>* log) : name_(name), log_(log) {}
  ~Obj() { log_->push_back("Deleting " + name_); }
  void f() const { log_->push_back("Obj::f() called on " + name_); }

 private:
  string name_;
  std::vector<string>* log_;
};

UniquePtrWithDeps<Obj> BuildObjWithDeps(std::vector<string>* log) {
  UniquePtrWithDeps<Obj> obj;
  auto dep1 = obj.AddDependency(MakeUnique<Dep>("dep1", log));
  auto dep2 = obj.AddDependency(MakeUnique<Dep>("dep2", log));
  auto dep3 = obj.AddDependency(MakeUnique<Dep>("dep3", log));
  dep1->f();
  dep2->f();
  dep3->f();
  obj.SetOwned(MakeUnique<Obj>("obj", log));
  return obj;
}

TEST(UniquePtrWithDepsTest, DependenciesDestroyedAfterMainObject) {
  std::vector<string> log;
  {
    auto obj = BuildObjWithDeps(&log);
    obj->f();
  }
  EXPECT_EQ(std::vector<string>({"dep1.f() called",
                            "dep2.f() called",
                            "dep3.f() called",
                            "Obj::f() called on obj",
                            "Deleting obj",
                            "Deleting dep3",
                            "Deleting dep2",
                            "Deleting dep1"}),
            log);
}

UniquePtrWithDeps<Obj> BuildObjWithDepAddedLater(std::vector<string>* log) {
  UniquePtrWithDeps<Obj> obj;
  obj.SetOwned(MakeUnique<Obj>("obj", log));
  auto dep = obj.AddDependency(MakeUnique<Dep>("dep", log));
  dep->f();
  return obj;
}

TEST(UniquePtrWithDepsTest, DependencyAddedAfterSetOwned) {
  std::vector<string> log;
  {
    auto obj = BuildObjWithDepAddedLater(&log);
    obj->f();
  }
  EXPECT_EQ(std::vector<string>({"dep.f() called",
                            "Obj::f() called on obj",
                            "Deleting dep",
                            "Deleting obj"}),
            log);
}

UniquePtrWithDeps<Obj> BuildObjWithDepsMultipleSetOwned(
    std::vector<string>* log) {
  UniquePtrWithDeps<Obj> obj;
  auto dep1 = obj.AddDependency(MakeUnique<Dep>("dep1", log));
  obj.SetOwned(MakeUnique<Obj>("obj1", log));
  auto dep2 = obj.AddDependency(MakeUnique<Dep>("dep2", log));
  auto dep3 = obj.AddDependency(MakeUnique<Dep>("dep3", log));
  dep1->f();
  dep2->f();
  dep3->f();
  obj.SetOwned(MakeUnique<Obj>("obj2", log));
  return obj;
}

TEST(UniquePtrWithDepsTest, MultipleSetOwnedCalls) {
  std::vector<string> log;
  {
    auto obj = BuildObjWithDepsMultipleSetOwned(&log);
    obj->f();
  }
  EXPECT_EQ(std::vector<string>({"dep1.f() called",
                            "dep2.f() called",
                            "dep3.f() called",
                            "Obj::f() called on obj2",
                            "Deleting obj2",
                            "Deleting dep3",
                            "Deleting dep2",
                            "Deleting obj1",
                            "Deleting dep1"}),
            log);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
