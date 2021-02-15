/* Copyright 2019 Google Inc. All Rights Reserved.

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

// Command line tool to create TF Lite Session from a TF Lite model file.

#include <iostream>
#include <memory>
#include <string>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow_serving/servables/tensorflow/tflite_session.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "ERROR: Missing filename. Usage: " << argv[0]
              << " <tflite-model-filename>" << std::endl;
    return 1;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  const std::string filename(argv[1]);
  std::string model_bytes;
  auto status =
      ReadFileToString(tensorflow::Env::Default(), filename, &model_bytes);
  if (!status.ok()) {
    std::cerr << "ERROR: Failed to read model file: " << filename
              << " with error: " << status << std::endl;
    return 1;
  }

  ::google::protobuf::Map<std::string, tensorflow::SignatureDef> signatures;
  std::unique_ptr<tensorflow::serving::TfLiteSession> session;
  tensorflow::SessionOptions options;
  status = tensorflow::serving::TfLiteSession::Create(
      std::move(model_bytes), options, /*num_pools=*/1,
      /*num_interpreters_per_pool=*/1, &session, &signatures);
  if (!status.ok()) {
    std::cerr << "ERROR: Failed to create TF Lite session with error: "
              << status << std::endl;
    return 1;
  }
  std::cout << "Successfully created TF Lite Session for model file: "
            << filename << std::endl;

  std::cout << "Signatures: " << std::endl;
  for (const auto& signature_info : signatures) {
    std::cout << "  " << signature_info.first << ": "
              << signature_info.second.DebugString() << std::endl;
  }
  return 0;
}
