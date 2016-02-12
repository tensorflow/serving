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

#include <stddef.h>
#include <memory>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace serving {
namespace {

using Hashmap = std::unordered_map<string, string>;

// Populates a hashmap from a file located at 'path', in format 'format'.
Status LoadHashmapFromFile(const string& path,
                           const HashmapSourceAdapterConfig::Format& format,
                           std::unique_ptr<Hashmap>* hashmap) {
  hashmap->reset(new Hashmap);
  switch (format) {
    case HashmapSourceAdapterConfig::SIMPLE_CSV: {
      RandomAccessFile* file;
      TF_RETURN_IF_ERROR(Env::Default()->NewRandomAccessFile(path, &file));
      const size_t kBufferSizeBytes = 262144;
      io::InputBuffer in(file, kBufferSizeBytes);
      string line;
      while (in.ReadLine(&line).ok()) {
        std::vector<string> cols = str_util::Split(line, ',');
        if (cols.size() != 2) {
          return errors::InvalidArgument("Unexpected format.");
        }
        const string& key = cols[0];
        const string& value = cols[1];
        (*hashmap)->insert({key, value});
      }
      break;
    }
    default:
      return errors::InvalidArgument("Unrecognized format enum value: ",
                                     format);
  }
  return Status::OK();
}

}  // namespace

HashmapSourceAdapter::HashmapSourceAdapter(
    const HashmapSourceAdapterConfig& config)
    : SimpleLoaderSourceAdapter<StoragePath, Hashmap>([config](
          const StoragePath& path, std::unique_ptr<Hashmap> * hashmap) {
        return LoadHashmapFromFile(path, config.format(), hashmap);
      }) {}

}  // namespace serving
}  // namespace tensorflow
