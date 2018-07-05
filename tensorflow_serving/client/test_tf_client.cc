#include "tf_client.h"
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
   ServingClient* client = create_client();
   client->Init("localhost:9000","lr","predict");
   std::vector<float> feats(243,1.0);
    std::cout << "feat size "<< feats.size();
    std::vector<size_t> shape({1,243});
    std::vector<float> score;
    std::cout << "Before predict \n";
   client->Predict(feats,shape,"input",score);
    std::cout << "After predict \n";
    for(auto i=score.begin(); i!=score.end(); i++)
    {
        std::cout<<*i<<" ";
    }
   return 0;
}
