#include <iostream>
#include <fstream>

#include <grpc++/create_channel.h>
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/command_line_flags.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;


using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map< std::string, tensorflow::TensorProto > OutMap;


class ServingClient {
 public:
  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {
  }

  std::string callPredict(std::string model_name, std::string file_path){
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(model_name);

    google::protobuf::Map< std::string, tensorflow::TensorProto >& inputs = 
        *predictRequest.mutable_inputs();

    tensorflow::TensorProto proto;

    std::ifstream imageFile(file_path, std::ios::binary);

    if (!imageFile.is_open()) {
      std::cout << "Failed to open " << file_path << std::endl;
      return "";
    }

    std::filebuf * pbuf = imageFile.rdbuf();
    long fileSize = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
    
    char* image = new char[fileSize]();

    pbuf->pubseekpos(0, std::ios::in);
    pbuf->sgetn(image, fileSize);
    imageFile.close();
    

    proto.set_dtype(tensorflow::DataType::DT_STRING);
    proto.add_string_val(image, fileSize);

    proto.mutable_tensor_shape()->add_dim()->set_size(1);

    inputs["images"] = proto;
    
    Status status = stub_->Predict(&context, predictRequest, &response);

    delete[] image;

    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is "<< response.outputs_size() << std::endl;
      OutMap& map_outputs =  *response.mutable_outputs();
      OutMap::iterator iter;
      int output_index = 0;
      
      for(iter = map_outputs.begin();iter != map_outputs.end(); ++iter){
        tensorflow::TensorProto& result_tensor_proto= iter->second;
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(result_tensor_proto);
        if (converted) {
          std::cout << "the result tensor[" << output_index << "] is:" <<
               std::endl << tensor.SummarizeValue(10) << std::endl;
        }else {
          std::cout << "the result tensor[" << output_index << 
               "] convert failed." << std::endl;
        }
        ++output_index;
      }
      return "Done.";
    } else {
      std::cout << "gRPC call return code: " 
          <<status.error_code() << ": " << status.error_message()
          << std::endl;
      return "gRPC failed.";
    }
  }

 private:
  std::unique_ptr<PredictionService::Stub> stub_;
};

int main(int argc, char** argv) {
  std::string server_port = "localhost:9000";
  std::string image_file = "";
  std::string model_name = "inception";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("server_port", &server_port, 
          "the IP and port of the server"),
      tensorflow::Flag("image_file", &image_file, 
          "the path to the "),
      tensorflow::Flag("model_name", &model_name, "name of model")
  };
  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || image_file.empty()) {
    std::cout << usage;
    return -1;
  }

  ServingClient guide(
      grpc::CreateChannel( server_port,
                          grpc::InsecureChannelCredentials()));
  std::cout << "calling predict using file: " << 
      image_file << "  ..." << std::endl;
  std::cout << guide.callPredict(model_name, image_file) << std::endl;

  return 0;
}
