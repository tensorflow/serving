#include "tf_client.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensor_util.h"
#include <iostream>
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

extern "C" ServingClient* create_client()
{
    return new ServingClient();
}

extern "C" void destroy_client(ServingClient* client)
{
    delete client;
}

ServingClient::ServingClient()
{
}

ServingClient::~ServingClient()
{
}

void ServingClient::Init(const std::string& server_port,const std::string& model_name, const std::string& signature)
{
    std::cout<<"Init 1\n";
    //std::shared_ptr<Channel> channel =  grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials());
    std::cout<<"Init 2\n";
    //this->stub_ = PredictionService::NewStub(channel).get();
    server_port_ = server_port;
    std::cout << "Stub "<<stub_<<"\n";
    std::cout<<"Init 3\n";
    model_name_ = model_name;
    std::cout<<"Init 4\n";
    signature_ = signature;
    std::cout<<"Init 5\n";
}


void ServingClient::Predict(const std::vector<float>& vals,const std::vector<size_t>& shape, const std::string& field_name, std::vector<float> &scores)
{
    std::cout << "Predict 1\n";
    scores.clear();
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    std::cout << "Predict 2\n";
    predictRequest.mutable_model_spec()->set_name(model_name_);
    predictRequest.mutable_model_spec()->set_signature_name(signature_);

    google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs =
        *predictRequest.mutable_inputs();

    std::shared_ptr<Channel> channel =  grpc::CreateChannel(server_port_, grpc::InsecureChannelCredentials());
    std::unique_ptr<PredictionService::Stub> stub(PredictionService::NewStub(channel));
    std::cout << "Predict 3\n";
    tensorflow::TensorProto proto = tensorflow::tensor::CreateTensorProto<float>(vals,shape);
    inputs[field_name] = proto;

    std::cout << "Predict 4 next is stub res\n";
    std::cout << "Predict 4 next is stub res\n";
    std::cout << "Predict 4 next is stub res\n";
    std::cout << "stub: \n";
    std::cout << "stub: \n";
    std::cout << "stub: \n";
    std::cout << "stub: \n";

    Status status = stub->Predict(&context, predictRequest, &response);

    std::cout << "Predict 5\n";
    if (status.ok()) {
        std::cout << "Predict 6\n";
        OutMap& map_outputs = *response.mutable_outputs();
        auto score_lst = map_outputs["output"].mutable_float_val();
        std::cout << "Predict 7\n";
        for(auto it=score_lst->begin(); it!=score_lst->end(); it++)
        {
            scores.push_back(*it);
        }
    }
    std::cout << "Predict 8\n";
}
