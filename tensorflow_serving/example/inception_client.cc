#include <iostream>
#include <fstream>

#include <grpc++/create_channel.h>
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"


using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;


using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

const std::string SERVING_HOST = "localhost:9000";

const std::string INPUT_FILE_NAME = "/tmp/cropped_panda.jpg";
const std::string SERVICE_NAME = "inception";


typedef ::google::protobuf::Map< ::std::string, ::tensorflow::TensorProto > OutMap;


class ServingClient {
 public:
  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {
  }

  std::string callPredict(std::string file_path){
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(SERVICE_NAME);

    ::google::protobuf::Map< ::std::string, ::tensorflow::TensorProto >& inputs = 
        *predictRequest.mutable_inputs();

    ::tensorflow::TensorProto proto;

    std::ifstream imageFile(file_path, std::ios::binary);

    if (!imageFile.is_open()) {
      std::cout << "Failed to open " << file_path << std::endl;
      return "";
    }

    std::filebuf * pbuf = imageFile.rdbuf();
    long fileSize = pbuf->pubseekoff(0, std::ios::end, std::ios::in);
    
    char* image = new char[fileSize];

    pbuf->pubseekpos(0, std::ios::in);
    pbuf->sgetn(image, fileSize);
    imageFile.close();
    

    proto.set_dtype(::tensorflow::DataType::DT_STRING);
    proto.add_string_val(image, fileSize);

    std::cout << "image size:" << fileSize << std::endl;
    proto.mutable_tensor_shape()->add_dim()->set_size(1);

    inputs["images"] = proto;
    
    Status status = stub_->Predict(&context, predictRequest, &response);
    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is "<< response.outputs_size() << std::endl;
      OutMap& map_outputs =  *response.mutable_outputs();
      OutMap::iterator iter;
      int output_index = 0;
      for(iter = map_outputs.begin();iter != map_outputs.end(); ++iter){
        std::cout << "output " << output_index << " name is "<< iter->first << std::endl;

        std::cout << "output " << output_index << " type is "<< 
            ::tensorflow::DataType_Name(
              iter->second.dtype()) << std::endl;

        int value_size =  0;
        switch(iter->second.dtype()){
          case ::tensorflow::DataType::DT_STRING:
            value_size = iter->second.string_val_size();
            break;
          case ::tensorflow::DataType::DT_FLOAT:
            value_size = iter->second.float_val_size();
            break;
          default:
            break;
        }
        std::cout << "outputs has "<< value_size << " value(s)" ;

        std::cout << " and shaped as [ " ;
        int dim_size = iter->second.tensor_shape().dim_size();
        ::google::protobuf::int64 * dims = 
            new ::google::protobuf::int64 [dim_size];
        for(int d=0; d < dim_size; ++d){
          dims[d] =  iter->second.tensor_shape().dim(d).size();
          std::cout << dims[d] << " ";
        }
        std::cout << "]" << std::endl;

        for(int k=0; k < value_size; ++k) {
          switch(iter->second.dtype()){
            case ::tensorflow::DataType::DT_STRING:
              std::cout << "value at " << k << " is :" << iter->second.string_val(k) << std::endl;
              break;
            case ::tensorflow::DataType::DT_FLOAT:
              std::cout << "value at " << k << " is :" << iter->second.float_val(k) << std::endl;
              break;
            default:
              break;
          }
        }  
         
        if(dims) {
          delete[] dims;
          dims = NULL;
        }
        ++output_index;
      }
      if(image) {
        delete[] image;
        image = NULL;
      }
      return "ok";
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      if(image) {
        delete[] image;
        image = NULL;
      }
      return "RPC failed";
    }
  }

 private:
  std::unique_ptr<PredictionService::Stub> stub_;
};


std::string GetImagePath(int argc, char** argv) {
  std::string image_path;
  std::string arg_str("--image");
  if (argc > 1) {
    std::string argv_1 = argv[1];
    size_t start_position = argv_1.find(arg_str);
    if (start_position != std::string::npos) {
      start_position += arg_str.size();
      if (argv_1[start_position] == ' ' ||
          argv_1[start_position] == '=') {
        image_path = argv_1.substr(start_position + 1);
      }
    }
  } else {
    image_path = INPUT_FILE_NAME;
  }
  return image_path;
}


int main(int argc, char** argv) {
  // Expect only arg: --db_path=path/to/route_guide_db.json.
  std::string image_path = GetImagePath(argc, argv);
  ServingClient guide(
      grpc::CreateChannel( SERVING_HOST,
                          grpc::InsecureChannelCredentials()));
  std::cout << "calling predict using file: " << image_path << "  ..." << std::endl;
  std::cout << guide.callPredict(image_path) << std::endl;

  return 0;
}
