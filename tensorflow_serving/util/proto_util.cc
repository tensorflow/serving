#include "tensorflow_serving/util/proto_util.h"

#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace serving {

tensorflow::Status ParseProtoTextFile(const string& file,
                                      google::protobuf::Message* message) {
  std::unique_ptr<tensorflow::ReadOnlyMemoryRegion> file_data;
  TF_RETURN_IF_ERROR(
          tensorflow::Env::Default()->NewReadOnlyMemoryRegionFromFile(file,
                                                                      &file_data));
  string file_data_str(static_cast<const char*>(file_data->data()),
                       file_data->length());
  if (tensorflow::protobuf::TextFormat::ParseFromString(file_data_str,
                                                        message)) {
    return tensorflow::Status::OK();
  } else {
    return tensorflow::errors::InvalidArgument("Invalid protobuf file: '", file,
                                               "'");
  }
}

/*template <typename ProtoType>
ProtoType ReadProtoFromFile(const string& file) {
  ProtoType proto;
  TF_CHECK_OK(ParseProtoTextFile(file, &proto));
  return proto;
}*/

} // namespace serving
} // namespace tensorflow
