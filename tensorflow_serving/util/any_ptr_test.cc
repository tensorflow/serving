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

#include "tensorflow_serving/util/any_ptr.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {
namespace {

TEST(AnyPtrTest, SetAndGet) {
  AnyPtr ptr;
  int object;

  // Implicitly settable/constructable from a raw pointer.
  ptr = &object;
  EXPECT_EQ(&object, ptr.get<int>());
  EXPECT_EQ(nullptr, ptr.get<bool>());

  // Implicitly settable/constructable from nullptr.
  ptr = nullptr;
  EXPECT_EQ(nullptr, ptr.get<int>());
  EXPECT_EQ(nullptr, ptr.get<bool>());
}

TEST(AnyPtrTest, ConstCorrect) {
  AnyPtr ptr;
  const int object = 0;
  ptr = &object;
  EXPECT_EQ(nullptr, ptr.get<int>());
  EXPECT_EQ(&object, ptr.get<const int>());
}

// Tests that a dynamic relationship between two classes doesn't cause any sort
// of type-punning.
TEST(AnyPtrTest, BaseClass) {
  class Base {
   public:
    virtual ~Base() {}

   private:
    int unused_base_var_ = 0;
  };

  class Child : public Base {
   public:
    ~Child() override {}

   private:
    int unused_child_var_ = 0;
  };

  AnyPtr ptr;
  Child c;
  ptr = &c;

  // Make sure casting to base returns null. This may work in some trivial
  // cases, but allowing down-casting in AnyPtr could break if, for example,
  // multiple inheretance is being used.
  EXPECT_EQ(nullptr, ptr.get<Base>());

  // Getting the pointer as the child type should work.
  EXPECT_EQ(&c, ptr.get<Child>());

  // Make sure accessing as base works if we store the pointer as the base
  // class.
  ptr = static_cast<Base*>(&c);
  EXPECT_EQ(&c, ptr.get<Base>());
  EXPECT_EQ(nullptr, ptr.get<Child>());
}

struct Destructable {
  ~Destructable() { *destroyed = true; }

  bool* const destroyed;
};

TEST(UniqueAnyPtrTest, SetGetAndDestroy) {
  bool destroyed = false;
  UniqueAnyPtr ptr;

  // Move constructable.
  ptr =
      UniqueAnyPtr{std::unique_ptr<Destructable>{new Destructable{&destroyed}}};
  EXPECT_EQ(&destroyed, ptr.get<Destructable>()->destroyed);
  EXPECT_EQ(nullptr, ptr.get<int>());
  ASSERT_FALSE(destroyed);

  // Implicitly settable/constructable from nullptr.
  ptr = nullptr;
  EXPECT_TRUE(destroyed);
}

TEST(UniqueAnyPtrTest, MoveConstruction) {
  UniqueAnyPtr ptr1 = UniqueAnyPtr(std::unique_ptr<int>(new int(1)));
  UniqueAnyPtr ptr2(std::move(ptr1));

  ASSERT_EQ(1, *ptr2.get<int>());
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
