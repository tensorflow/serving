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

#include "tensorflow_serving/util/cleanup.h"

#include <functional>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

namespace tensorflow {
namespace serving {

using AnyCleanup = Cleanup<std::function<void()>>;

template <typename T1, typename T2>
void AssertTypeEq() {
  static_assert(std::is_same<T1, T2>::value, "unexpected type");
}

TEST(CleanupTest, BasicLambda) {
  string s = "active";
  {
    auto s_cleaner = MakeCleanup([&s] { s.assign("cleaned"); });
    EXPECT_EQ("active", s);
  }
  EXPECT_EQ("cleaned", s);
}

TEST(FinallyTest, NoCaptureLambda) {
  // Noncapturing lambdas are just structs and use aggregate initializers.
  // Make sure MakeCleanup is compatible with that kind of initialization.
  static string& s = *new string;
  s.assign("active");
  {
    auto s_cleaner = MakeCleanup([] { s.append(" clean"); });
    EXPECT_EQ("active", s);
  }
  EXPECT_EQ("active clean", s);
}

TEST(CleanupTest, Release) {
  string s = "active";
  {
    auto s_cleaner = MakeCleanup([&s] { s.assign("cleaned"); });
    EXPECT_EQ("active", s);
    s_cleaner.release();
  }
  EXPECT_EQ("active", s);  // no cleanup should have occurred.
}

TEST(FinallyTest, TypeErasedWithoutFactory) {
  string s = "active";
  {
    AnyCleanup s_cleaner([&s] { s.append(" clean"); });
    EXPECT_EQ("active", s);
  }
  EXPECT_EQ("active clean", s);
}

struct Appender {
  Appender(string* s_in, const string& msg_in) : s(s_in), msg(msg_in) {}
  void operator()() const { s->append(msg); }
  string* s;
  string msg;
};

TEST(CleanupTest, NonLambda) {
  string s = "active";
  {
    auto c = MakeCleanup(Appender(&s, " cleaned"));
    AssertTypeEq<decltype(c), Cleanup<Appender>>();
    EXPECT_EQ("active", s);
  }
  EXPECT_EQ("active cleaned", s);
}

TEST(CleanupTest, Assign) {
  string s = "0";
  {
    auto clean1 = MakeCleanup(Appender(&s, " 1"));
    auto clean2 = MakeCleanup(Appender(&s, " 2"));
    EXPECT_EQ("0", s);
    clean2 = std::move(clean1);
    EXPECT_EQ("0 2", s);
  }
  EXPECT_EQ("0 2 1", s);
}

TEST(CleanupTest, AssignAny) {
  // Check that implicit conversions can happen in assignment.
  string s = "0";
  {
    auto clean1 = MakeCleanup(Appender(&s, " 1"));
    AnyCleanup clean2 = MakeCleanup(Appender(&s, " 2"));
    EXPECT_EQ("0", s);
    clean2 = std::move(clean1);
    EXPECT_EQ("0 2", s);
  }
  EXPECT_EQ("0 2 1", s);
}

TEST(CleanupTest, AssignFromReleased) {
  string s = "0";
  {
    auto clean1 = MakeCleanup(Appender(&s, " 1"));
    auto clean2 = MakeCleanup(Appender(&s, " 2"));
    EXPECT_EQ("0", s);
    clean1.release();
    clean2 = std::move(clean1);
    EXPECT_EQ("0 2", s);
  }
  EXPECT_EQ("0 2", s);
}

TEST(CleanupTest, AssignToReleased) {
  string s = "0";
  {
    auto clean1 = MakeCleanup(Appender(&s, " 1"));
    auto clean2 = MakeCleanup(Appender(&s, " 2"));
    EXPECT_EQ("0", s);
    clean2.release();
    EXPECT_EQ("0", s);
    clean2 = std::move(clean1);
    EXPECT_EQ("0", s);
  }
  EXPECT_EQ("0 1", s);
}

TEST(CleanupTest, AssignToDefaultInitialized) {
  string s = "0";
  {
    auto clean1 = MakeCleanup(Appender(&s, " 1"));
    {
      AnyCleanup clean2;
      EXPECT_EQ("0", s);
      clean2 = std::move(clean1);
      EXPECT_EQ("0", s);
    }
    EXPECT_EQ("0 1", s);
  }
  EXPECT_EQ("0 1", s);
}

class CleanupReferenceTest : public testing::Test {
 public:
  struct F {
    int* cp;
    int* i;
    F(int* cp, int* i) : cp(cp), i(i) {}
    F(const F& o) : cp(o.cp), i(o.i) { ++*cp; }
    F& operator=(const F& o) {
      cp = o.cp;
      i = o.i;
      ++*cp;
      return *this;
    }
    F(F&&) = default;
    F& operator=(F&&) = default;
    void operator()() const { ++*i; }
  };
  int copies_ = 0;
  int calls_ = 0;
  F f_ = F(&copies_, &calls_);

  static int g_calls;
  void SetUp() override { g_calls = 0; }
  static void CleanerFunction() { ++g_calls; }
};
int CleanupReferenceTest::g_calls = 0;

TEST_F(CleanupReferenceTest, FunctionPointer) {
  {
    auto c = MakeCleanup(&CleanerFunction);
    AssertTypeEq<decltype(c), Cleanup<void (*)()>>();
    EXPECT_EQ(0, g_calls);
  }
  EXPECT_EQ(1, g_calls);
  // Test that a function reference decays to a function pointer.
  {
    auto c = MakeCleanup(CleanerFunction);
    AssertTypeEq<decltype(c), Cleanup<void (*)()>>();
    EXPECT_EQ(1, g_calls);
  }
  EXPECT_EQ(2, g_calls);
}

TEST_F(CleanupReferenceTest, AssignLvalue) {
  string s = "0";
  Appender app1(&s, "1");
  Appender app2(&s, "2");
  {
    auto c = MakeCleanup(app1);
    c.release();
    c = MakeCleanup(app2);
    EXPECT_EQ("0", s);
    app1();
    EXPECT_EQ("01", s);
  }
  EXPECT_EQ("012", s);
}

TEST_F(CleanupReferenceTest, FunctorLvalue) {
  // Test that MakeCleanup(lvalue) produces Cleanup<F>, not Cleanup<F&>.
  EXPECT_EQ(0, copies_);
  EXPECT_EQ(0, calls_);
  {
    auto c = MakeCleanup(f_);
    AssertTypeEq<decltype(c), Cleanup<F>>();
    EXPECT_EQ(1, copies_);
    EXPECT_EQ(0, calls_);
  }
  EXPECT_EQ(1, copies_);
  EXPECT_EQ(1, calls_);
  {
    auto c = MakeCleanup(f_);
    EXPECT_EQ(2, copies_);
    EXPECT_EQ(1, calls_);
    F f2 = c.release();  // release is a move.
    EXPECT_EQ(2, copies_);
    EXPECT_EQ(1, calls_);
    auto c2 = MakeCleanup(f2);  // copy
    EXPECT_EQ(3, copies_);
    EXPECT_EQ(1, calls_);
  }
  EXPECT_EQ(3, copies_);
  EXPECT_EQ(2, calls_);
}

TEST_F(CleanupReferenceTest, FunctorRvalue) {
  {
    auto c = MakeCleanup(std::move(f_));
    AssertTypeEq<decltype(c), Cleanup<F>>();
    EXPECT_EQ(0, copies_);
    EXPECT_EQ(0, calls_);
  }
  EXPECT_EQ(0, copies_);
  EXPECT_EQ(1, calls_);
}

TEST_F(CleanupReferenceTest, FunctorReferenceWrapper) {
  {
    auto c = MakeCleanup(std::cref(f_));
    AssertTypeEq<decltype(c), Cleanup<std::reference_wrapper<const F>>>();
    EXPECT_EQ(0, copies_);
    EXPECT_EQ(0, calls_);
  }
  EXPECT_EQ(0, copies_);
  EXPECT_EQ(1, calls_);
}

}  // namespace serving
}  // namespace tensorflow
