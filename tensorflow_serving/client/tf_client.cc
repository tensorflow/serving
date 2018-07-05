#include "tf_client.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"
//#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/platform/types.h"
//#include "tensorflow/core/util/command_line_flags.h"
#include "prediction_service.grpc.pb.h"
//#include "tensor_util.h"
#include <iostream>
#include <glog/logging.h>
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

//typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

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
    //std::shared_ptr<Channel> channel =  grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials());
    //this->stub_ = PredictionService::NewStub(channel).get();
    server_port_ = server_port;
    model_name_ = model_name;
    signature_ = signature;
    LOG(INFO) << "TensorflowClient Configured. Server: "<<server_port<< " model "<<model_name<< "sigature "<< signature;
}

void fill_values(tensorflow::TensorProto& proto,const std::vector<float> & vals, const std::vector<size_t>& shape)
{
    for(auto val : vals) proto.mutable_float_val()->Add(val);
    for (auto dim : shape) {
        proto.mutable_tensor_shape()->mutable_dim()->Add()->set_size(dim);
    }
}

void ServingClient::Predict(const std::vector<float>& vals,const std::vector<size_t>& shape, const std::string& field_name, std::vector<float> &scores)
{
    scores.clear();
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(model_name_);
    predictRequest.mutable_model_spec()->set_signature_name(signature_);

    //google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs =
    std::shared_ptr<Channel> channel =  grpc::CreateChannel(server_port_, grpc::InsecureChannelCredentials());
    std::unique_ptr<PredictionService::Stub> stub(PredictionService::NewStub(channel));
    tensorflow::TensorProto proto;// = tensorflow::tensor::CreateTensorProto<float>(vals,shape);
    proto.set_dtype(tensorflow::DataType::DT_FLOAT);
    fill_values(proto,vals,shape);
    (*predictRequest.mutable_inputs())[field_name] = proto;
    Status status = stub->Predict(&context, predictRequest, &response);

    if (status.ok()) {
        auto map_outputs = *response.mutable_outputs();
        auto score_lst = map_outputs["output"].mutable_float_val();
        for(auto it=score_lst->begin(); it!=score_lst->end(); it++)
        {
            scores.push_back(*it);
        }
        LOG(INFO) << " Tensorflow service responded. ";
    }
    else {
        LOG(ERROR) << "Failed to call tensorflow service Err msg "<< status.error_message() << " Err code "<<status.error_code();
    }
}
