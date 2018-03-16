cc_library(
    name = "model_selector",
    srcs = ["model_selector.cc"],
    hdrs = ["model_selector.h"],
    deps = [
        ":model_selector_proto",
        "//tensorflow_serving/apis:model_proto",
        "//tensorflow_serving/servables/tensorflow:util",
    ],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "model_selector_source_adapter",
    srcs = ["model_selector_source_adapter.cc"],
    hdrs = ["model_selector_source_adapter.h"],
    deps = [
        ":model_selector",
        ":model_selector_source_adapter_proto",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/util:proto_util",
    ],
    visibility = [
        "//visibility:public",
    ],
    alwayslink = 1,
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "model_selector_proto",
    srcs = ["model_selector.proto"],
    deps = [
        "//tensorflow_serving/apis:model_proto",
        "@protobuf_archive//:cc_wkt_protos",
    ],
    cc_api_version = 2,
)

serving_proto_library(
    name = "model_selector_source_adapter_proto",
    srcs = ["model_selector_source_adapter.proto"],
    visibility = [
        "//visibility:public",
    ],
    cc_api_version = 2,
)
