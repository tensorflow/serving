load("@protobuf//:protobuf.bzl", "cc_proto_library")
load("@protobuf//:protobuf.bzl", "py_proto_library")

def serving_proto_library(name, srcs=[], has_services=False,
                          deps=[], visibility=None, testonly=0,  # pylint: disable=unused-argument
                          cc_grpc_version = None,
                          cc_api_version=2, go_api_version=2,
                          java_api_version=2,
                          py_api_version=2):
  native.filegroup(name=name + "_proto_srcs",
                   srcs=srcs,
                   testonly=testonly,)

  use_grpc_plugin = None
  if cc_grpc_version:
    use_grpc_plugin = True
  cc_proto_library(name=name,
                   srcs=srcs,
                   deps=deps,
                   cc_libs = ["@protobuf//:protobuf"],
                   protoc="@protobuf//:protoc",
                   default_runtime="@protobuf//:protobuf",
                   use_grpc_plugin=use_grpc_plugin,
                   testonly=testonly,
                   visibility=visibility,)

def serving_go_grpc_library(**kwargs):  # pylint: disable=unused-argument
  """Build the Go gRPC bindings for a service. Not yet implemented."""
  return

def serving_proto_library_py(name, proto_library, srcs=[], deps=[], visibility=None, testonly=0):  # pylint: disable=unused-argument
  py_proto_library(name=name,
                   srcs=srcs,
                   srcs_version = "PY2AND3",
                   deps=["@protobuf//:protobuf_python"] + deps,
                   default_runtime="@protobuf//:protobuf_python",
                   protoc="@protobuf//:protoc",
                   visibility=visibility,
                   testonly=testonly,)
