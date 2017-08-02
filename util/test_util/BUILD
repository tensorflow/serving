# Description: testing utils for Tensorflow Serving utils.

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "mock_file_probing_env",
    testonly = 1,
    hdrs = ["mock_file_probing_env.h"],
    deps = [
        "//external:gtest",
        "//tensorflow_serving/util:file_probing_env",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
