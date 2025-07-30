load("//tensorflow_serving:serving.bzl", "if_google", "if_oss")

licenses(["notice"])

package(default_visibility = [
    "//tensorflow_serving:internal",
])

filegroup(
    name = "session_bundle_half_plus_two",
    srcs = glob([
        "testdata/half_plus_two/**",
    ]),
)

cc_library(
    name = "manifest_proto_header",
    hdrs = ["manifest_proto.h"],
    deps = ["//tensorflow_serving/util:oss_or_google"] + if_oss([
        "//tensorflow_serving/session_bundle/oss:manifest_proto_cc",
    ]) + if_google(["//tensorflow_serving/session_bundle/google:manifest_proto_cc"]),
)

cc_library(
    name = "session_bundle",
    hdrs = ["session_bundle.h"],
    deps = ["//tensorflow_serving/util:oss_or_google"] + if_oss([
        "//tensorflow_serving/session_bundle/oss:session_bundle",
    ]) + if_google(["@org_tensorflow//tensorflow/contrib/session_bundle"]),
)

cc_library(
    name = "session_bundle_util_header",
    hdrs = ["session_bundle_util.h"],
    deps = [
        ":manifest_proto_header",
        ":session_bundle",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
    ],
)

cc_library(
    name = "session_bundle_util",
    deps = if_oss(["//tensorflow_serving/session_bundle/oss:session_bundle_util"]) + if_google(["//tensorflow_serving/session_bundle/google:session_bundle_util"]),
)

cc_library(
    name = "graph_rewriter",
    hdrs = ["graph_rewriter.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:errors",
        "@org_tensorflow//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "saved_model_config",
    srcs = ["saved_model_config.cc"],
    hdrs = ["saved_model_config.h"],
    deps = [
        "//tensorflow_serving/servables/tensorflow:saved_model_config_cc_proto",
        "//tensorflow_serving/servables/tensorflow:saved_model_config_util",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "saved_model_config_test",
    srcs = ["saved_model_config_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_cpu_with_empty_saved_model_config",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_cpu_with_saved_model_config",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":saved_model_config",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/servables/tensorflow:remote_op_config_rewriter_cc_proto",
        "//tensorflow_serving/servables/tensorflow:saved_model_config_cc_proto",
        "//tensorflow_serving/servables/tensorflow:saved_model_config_util",
        "//tensorflow_serving/test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/grappler/optimizers/inference:batch_op_rewriter_proto_cc",
    ],
)
