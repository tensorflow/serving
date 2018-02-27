# Description: Tensorflow Serving batching test utilities.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_library(
    name = "puppet_batch_scheduler",
    testonly = 1,
    hdrs = ["puppet_batch_scheduler.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@org_tensorflow//tensorflow/contrib/batching:batch_scheduler",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "puppet_batch_scheduler_test",
    srcs = [
        "puppet_batch_scheduler_test.cc",
    ],
    deps = [
        ":puppet_batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

# script that generates saved_model for matrix_half_plus_two model.
py_binary(
    name = "matrix_half_plus_two_saved_model",
    srcs = ["matrix_half_plus_two_saved_model.py"],
    srcs_version = "PY2AND3",
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)
