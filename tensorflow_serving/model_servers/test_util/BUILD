# Description: Model Server test utility

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
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
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/apis:model_proto",
        "//tensorflow_serving/config:model_server_config_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_state",
        "//tensorflow_serving/core/test_util:fake_loader_source_adapter",
        "//tensorflow_serving/core/test_util:fake_loader_source_adapter_proto",
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
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export.index",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export.meta",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/core/test_util:fake_loader_source_adapter",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "storage_path_error_injecting_source_adapter",
    testonly = 1,
    srcs = ["storage_path_error_injecting_source_adapter.cc"],
    hdrs = ["storage_path_error_injecting_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":storage_path_error_injecting_source_adapter_proto",
        "//tensorflow_serving/core:source_adapter",
    ],
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "storage_path_error_injecting_source_adapter_proto",
    testonly = 1,
    srcs = ["storage_path_error_injecting_source_adapter.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
)
