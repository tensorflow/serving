# Description: Model Server test utility

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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
    name = "mock_server_core",
    testonly = 1,
    hdrs = ["mock_server_core.h"],
    deps = [
        "//tensorflow_serving/apis:model_proto",
        "//tensorflow_serving/config:model_server_config_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_state",
        "//tensorflow_serving/model_servers:server_core",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "server_core_test_util",
    testonly = 1,
    srcs = ["server_core_test_util.cc"],
    hdrs = ["server_core_test_util.h"],
    deps = [
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/model_servers:server_core",
    ],
)
