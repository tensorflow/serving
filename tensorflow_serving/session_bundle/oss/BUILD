load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow_serving:oss_or_google.bzl", "oss_only_cc_test")

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

tf_proto_library(
    name = "manifest_proto",
    srcs = ["manifest.proto"],
    cc_api_version = 2,
)

cc_library(
    name = "session_bundle",
    hdrs = ["session_bundle.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
)

cc_library(
    name = "session_bundle_util",
    srcs = ["session_bundle_util.cc"],
    deps = [
        ":manifest_proto_cc",
        ":session_bundle",
        "//tensorflow_serving/session_bundle:session_bundle_util_header",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

oss_only_cc_test(
    name = "session_bundle_util_test",
    srcs = ["session_bundle_util_test.cc"],
    data = [
        "//tensorflow_serving/session_bundle:session_bundle_half_plus_two",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":session_bundle_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
    ],
)
