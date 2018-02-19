#ifndef TENSORFLOW_SERVING_UTIL_PROTO_UTIL_H_
#define TENSORFLOW_SERVING_UTIL_PROTO_UTIL_H_

#include <google/protobuf/message.h>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

tensorflow::Status ParseProtoTextFile(const string& file,
                                      google::protobuf::Message* message);

template <typename ProtoType>
ProtoType ReadProtoFromFile(const string& file) {
  ProtoType proto;
  TF_CHECK_OK(ParseProtoTextFile(file, &proto));
  return proto;
}

}
}

#endif // TENSORFLOW_SERVING_UTIL_PROTO_UTIL_H_
