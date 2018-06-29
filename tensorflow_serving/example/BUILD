package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

cc_library(name="tfclient",
           srcs=["tf_client.cc"],
           hdrs=["tf_client.h","tensor_util.h"],
           deps = [
            "//tensorflow_serving/apis:prediction_service_proto",
            "@grpc//:grpc++_unsecure",
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:lib",
            "@protobuf_archive//:protobuf_lite",
            ],
          )
