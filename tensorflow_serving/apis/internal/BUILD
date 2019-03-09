# Internal implementation details of serving APIs.

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "serialized_input_proto",
    srcs = ["serialized_input.proto"],
    cc_api_version = 2,
    visibility = [
        "//tensorflow_serving:internal",
        "@org_tensorflow//tensorflow_ranking/google:__pkg__",
    ],
    deps = [
    ],
)
