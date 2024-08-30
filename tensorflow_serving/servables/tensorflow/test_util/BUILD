# Description: Tensorflow servables test_utils.

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

licenses(["notice"])

serving_proto_library(
    name = "fake_thread_pool_factory_proto",
    testonly = 1,
    srcs = ["fake_thread_pool_factory.proto"],
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

cc_library(
    name = "mock_tfrt_saved_model",
    testonly = 1,
    hdrs = ["mock_tfrt_saved_model.h"],
    visibility = [
        "//tensorflow_serving/batching:__subpackages__",
        "//tensorflow_serving/servables/tensorflow:__subpackages__",
    ],
    deps = [
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core/tfrt/runtime",
        "@org_tensorflow//tensorflow/core/tfrt/saved_model",
    ],
)
