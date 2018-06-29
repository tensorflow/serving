#include "tf_client.h"
extern "C" ServingClient* create_client()
{
    return new ServingClient();
} 


