#ifndef _SERVING_CLIENT_H
#define _SERVING_CLIENT_H
#include <vector>
#include <string>

class ServingClient{
public:

    ServingClient();

    virtual ~ServingClient();

    virtual void Predict(const std::vector<float>& vals,const std::vector<size_t>& shape, const std::string& field_name, std::vector<float>& scores);
    virtual void Init(const std::string& host,const std::string& model_name, const std::string& signature);
private:
    void *stub_;
    std::string model_name_;
    std::string signature_;
    std::string server_port_;
};

extern "C" ServingClient* create_client();
extern "C" void destroy_client(ServingClient* client);
#endif
