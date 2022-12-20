# Description: Tensorflow servables test_utils.

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

licenses(["notice"])

serving_proto_library(
    name = "fake_thread_pool_factory_proto",
    testonly = 1,
    srcs = ["fake_thread_pool_factory.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "fake_thread_pool_factory",
    testonly = 1,
    srcs = ["fake_thread_pool_factory.cc"],
    hdrs = ["fake_thread_pool_factory.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":fake_thread_pool_factory_cc_proto",
        "//tensorflow_serving/servables/tensorflow:thread_pool_factory",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
