load("//tensorflow_serving:oss_or_google.bzl", "oss_only_cc_test")

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

oss_only_cc_test(
    name = "session_bundle_source_adapter_test",
    size = "medium",
    srcs = ["session_bundle_source_adapter_test.cc"],
    data = ["//tensorflow_serving/session_bundle:session_bundle_half_plus_two"],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resources_proto",
        "//tensorflow_serving/servables/tensorflow:bundle_factory_test_util",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter",
        "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
