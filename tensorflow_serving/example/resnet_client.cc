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

#include <setjmp.h>

#include <fstream>
#include <iostream>

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/jpeg.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

struct tf_jpeg_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

typedef struct tf_jpeg_error_mgr* tf_jpeg_error_ptr;

METHODDEF(void)
tf_jpeg_error_exit(j_common_ptr cinfo) {
  tf_jpeg_error_ptr tf_jpeg_err = (tf_jpeg_error_ptr)cinfo->err;

  (*cinfo->err->output_message)(cinfo);

  longjmp(tf_jpeg_err->setjmp_buffer, 1);
}

class ServingClient {
 public:
  // JPEG decompression code following libjpeg-turbo documentation:
  // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/example.txt
  int readJPEG(const char* file_name, tensorflow::TensorProto* proto) {
    struct tf_jpeg_error_mgr jerr;
    FILE* infile;
    JSAMPARRAY buffer;
    int row_stride;
    struct jpeg_decompress_struct cinfo;

    if ((infile = fopen(file_name, "rb")) == NULL) {
      fprintf(stderr, "can't open %s\n", file_name);
      return -1;
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = tf_jpeg_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
      jpeg_destroy_decompress(&cinfo);
      fclose(infile);
      return -1;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    (void)jpeg_read_header(&cinfo, TRUE);

    (void)jpeg_start_decompress(&cinfo);
    row_stride = cinfo.output_width * cinfo.output_components;
    CHECK(cinfo.output_components == 3)
        << "Only 3-channel (RGB) JPEG files are supported";

    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                        row_stride, 1);

    proto->set_dtype(tensorflow::DataType::DT_FLOAT);
    while (cinfo.output_scanline < cinfo.output_height) {
      (void)jpeg_read_scanlines(&cinfo, buffer, 1);
      for (size_t i = 0; i < cinfo.output_width; i++) {
        proto->add_float_val(buffer[0][i * 3] / 255.0);
        proto->add_float_val(buffer[0][i * 3 + 1] / 255.0);
        proto->add_float_val(buffer[0][i * 3 + 2] / 255.0);
      }
    }

    proto->mutable_tensor_shape()->add_dim()->set_size(1);
    proto->mutable_tensor_shape()->add_dim()->set_size(cinfo.output_height);
    proto->mutable_tensor_shape()->add_dim()->set_size(cinfo.output_width);
    proto->mutable_tensor_shape()->add_dim()->set_size(cinfo.output_components);

    (void)jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 0;
  }

  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {}

  tensorflow::string callPredict(const tensorflow::string& model_name,
                                 const tensorflow::string& model_signature_name,
                                 const tensorflow::string& file_path) {
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(model_name);
    predictRequest.mutable_model_spec()->set_signature_name(
        model_signature_name);

    google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs =
        *predictRequest.mutable_inputs();

    tensorflow::TensorProto proto;

    const char* infile = file_path.c_str();

    if (readJPEG(infile, &proto)) {
      std::cout << "error constructing the protobuf";
      return "execution failed";
    }

    inputs["input_1"] = proto;

    Status status = stub_->Predict(&context, predictRequest, &response);

    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is " << response.outputs_size() << std::endl;
      OutMap& map_outputs = *response.mutable_outputs();
      OutMap::iterator iter;
      int output_index = 0;

      for (iter = map_outputs.begin(); iter != map_outputs.end(); ++iter) {
        tensorflow::TensorProto& result_tensor_proto = iter->second;
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(result_tensor_proto);
        if (converted) {
          std::cout << "the result tensor[" << output_index
                    << "] is:" << std::endl
                    << tensor.SummarizeValue(1001) << std::endl;
        } else {
          std::cout << "the result tensor[" << output_index
                    << "] convert failed." << std::endl;
        }
        ++output_index;
      }
      return "Done.";
    } else {
      std::cout << "gRPC call return code: " << status.error_code() << ": "
                << status.error_message() << std::endl;
      return "gRPC failed.";
    }
  }

 private:
  std::unique_ptr<PredictionService::Stub> stub_;
};

int main(int argc, char** argv) {
  tensorflow::string server_port = "localhost:8500";
  tensorflow::string image_file = "";
  tensorflow::string model_name = "resnet";
  tensorflow::string model_signature_name = "serving_default";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("server_port", &server_port,
                       "the IP and port of the server"),
      tensorflow::Flag("image_file", &image_file, "the path to the image"),
      tensorflow::Flag("model_name", &model_name, "name of model"),
      tensorflow::Flag("model_signature_name", &model_signature_name,
                       "name of model signature")};

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || image_file.empty()) {
    std::cout << usage;
    return -1;
  }

  ServingClient guide(
      grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials()));
  std::cout << "calling predict using file: " << image_file << "  ..."
            << std::endl;
  std::cout << guide.callPredict(model_name, model_signature_name, image_file)
            << std::endl;
  return 0;
}
