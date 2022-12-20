# Description: Model Server test utility

load("//tensorflow_serving:oss_or_google.bzl", "if_google")
load("//tensorflow_serving:serving.bzl", "serving_proto_library")

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])

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
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/config:model_server_config_cc_proto",
        "//tensorflow_serving/config:platform_config_cc_proto",
        "//tensorflow_serving/core:aspired_versions_manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_state",
        "//tensorflow_serving/core:servable_state_monitor",
        "//tensorflow_serving/core:server_request_logger",
        "//tensorflow_serving/core/test_util:fake_loader_source_adapter",
        "//tensorflow_serving/core/test_util:fake_loader_source_adapter_cc_proto",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/util:event_bus",
        "//tensorflow_serving/util:unique_ptr_with_deps",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "server_core_test_util",
    testonly = 1,
    srcs = ["server_core_test_util.cc"],
    hdrs = ["server_core_test_util.h"],
    data = [
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_2_versions",
    ] + if_google([
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.index",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.meta",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two_2_versions/00000123/export.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two_2_versions/00000123/export.index",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two_2_versions/00000123/export.meta",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two_2_versions/00000124/export.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two_2_versions/00000124/export.index",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two_2_versions/00000124/export.meta",
    ]),
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
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter_cc_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_cc_proto",
        "//tensorflow_serving/test_util",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core:lib",
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
        ":storage_path_error_injecting_source_adapter_cc_proto",
        "//tensorflow_serving/core:source_adapter",
    ],
)

serving_proto_library(
    name = "storage_path_error_injecting_source_adapter_proto",
    testonly = 1,
    srcs = ["storage_path_error_injecting_source_adapter.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "tensorflow_model_server_test_base",
    srcs = ["tensorflow_model_server_test_base.py"],
    srcs_version = "PY2AND3",
    tags = [
        "noasan",  # b/136281879
        "notsan",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "@org_tensorflow//tensorflow/python:platform",
        "@org_tensorflow//tensorflow/python/saved_model:signature_constants",
        "@six_archive//:six",
    ],
)
