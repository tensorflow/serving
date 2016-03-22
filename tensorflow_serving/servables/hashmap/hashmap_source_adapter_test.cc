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

#include "tensorflow_serving/servables/hashmap/hashmap_source_adapter.h"

#include <memory>
#include <unordered_map>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/servable_data.h"
#include "tensorflow_serving/core/test_util/source_adapter_test_util.h"
#include "tensorflow_serving/servables/hashmap/hashmap_source_adapter.pb.h"
#include "tensorflow_serving/util/any_ptr.h"

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace tensorflow {
namespace serving {
namespace {

using Hashmap = std::unordered_map<string, string>;

// Writes the given hashmap to a file.
Status WriteHashmapToFile(const HashmapSourceAdapterConfig::Format format,
                          const string& file_name, const Hashmap& hashmap) {
  WritableFile* file_raw;
  TF_RETURN_IF_ERROR(Env::Default()->NewWritableFile(file_name, &file_raw));
  std::unique_ptr<WritableFile> file(file_raw);
  switch (format) {
    case HashmapSourceAdapterConfig::SIMPLE_CSV: {
      for (const auto& entry : hashmap) {
        const string& key = entry.first;
        const string& value = entry.second;
        const string line = strings::StrCat(key, ",", value, "\n");
        file->Append(line);
      }
      break;
    }
    default:
      return errors::InvalidArgument("Unrecognized format enum value: ",
                                     format);
  }
  TF_RETURN_IF_ERROR(file->Close());
  return Status::OK();
}

TEST(HashmapSourceAdapter, Basic) {
  const auto format = HashmapSourceAdapterConfig::SIMPLE_CSV;
  const string file = io::JoinPath(testing::TmpDir(), "Basic");
  TF_ASSERT_OK(
      WriteHashmapToFile(format, file, {{"a", "apple"}, {"b", "banana"}}));

  HashmapSourceAdapterConfig config;
  config.set_format(format);
  auto adapter =
      std::unique_ptr<HashmapSourceAdapter>(new HashmapSourceAdapter(config));
  ServableData<std::unique_ptr<Loader>> loader_data =
      test_util::RunSourceAdapter(file, adapter.get());
  TF_ASSERT_OK(loader_data.status());
  std::unique_ptr<Loader> loader = loader_data.ConsumeDataOrDie();

  TF_ASSERT_OK(loader->Load(ResourceAllocation()));

  const Hashmap* hashmap = loader->servable().get<Hashmap>();
  EXPECT_THAT(*hashmap,
              UnorderedElementsAre(Pair("a", "apple"), Pair("b", "banana")));

  loader->Unload();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
