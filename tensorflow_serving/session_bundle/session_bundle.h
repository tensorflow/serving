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

// Low-level functionality for setting up a inference Session.

#ifndef TENSORFLOW_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_H_
#define TENSORFLOW_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_H_

#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow_serving/session_bundle/manifest.pb.h"
#include "tensorflow_serving/session_bundle/signature.h"

namespace tensorflow {
namespace serving {

const char kMetaGraphDefFilename[] = "export.meta";
const char kAssetsDirectory[] = "assets";
const char kInitOpKey[] = "serving_init_op";
const char kAssetsKey[] = "serving_assets";
const char kGraphKey[] = "serving_graph";

// Data and objects loaded from a python Exporter export.
// WARNING(break-tutorial-inline-code): The following code snippet is
// in-lined in tutorials, please update tutorial documents accordingly
// whenever code changes.
struct SessionBundle {
  std::unique_ptr<tensorflow::Session> session;
  tensorflow::MetaGraphDef meta_graph_def;
};

// Loads a manifest and initialized session using the output of an Exporter
// using the format defined at https://goo.gl/OIDCqz.
tensorflow::Status LoadSessionBundleFromPath(
    const tensorflow::SessionOptions& options,
    const tensorflow::StringPiece export_dir, SessionBundle* bundle);

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_SESSION_BUNDLE_SESSION_BUNDLE_H_
