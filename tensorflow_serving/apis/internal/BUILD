# Internal implementation details of serving APIs.

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

package(features = ["-layering_check"])

licenses(["notice"])

serving_proto_library(
    name = "serialized_input_proto",
    srcs = ["serialized_input.proto"],
    visibility = [
        "//tensorflow_serving:internal",
        "@org_tensorflow//tensorflow_ranking/google:__pkg__",
    ],
    deps = [
    ],
)
