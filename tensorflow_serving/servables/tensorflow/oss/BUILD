package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])

cc_library(
    name = "resource_estimator",
    srcs = ["resource_estimator.cc"],
    hdrs = ["resource_estimator.h"],
    deps = [
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_cc_proto",
        "//tensorflow_serving/servables/tensorflow:util",
        "//tensorflow_serving/util:file_probing_env",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "resource_estimator_test",
    srcs = ["resource_estimator_test.cc"],
    deps = [
        ":resource_estimator",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/servables/tensorflow:bundle_factory_test_util",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util/test_util:mock_file_probing_env",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)
