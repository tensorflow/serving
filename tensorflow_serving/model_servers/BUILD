# Description: Model Server

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
    name = "server_core",
    srcs = ["server_core.cc"],
    hdrs = ["server_core.h"],
    visibility = [
        "//tensorflow_serving:internal",
    ],
    deps = [
        "//tensorflow_serving/apis:model_proto",
        "//tensorflow_serving/config:model_server_config_proto",
        "//tensorflow_serving/core:aspired_versions_manager",
        "//tensorflow_serving/core:eager_load_policy",
        "//tensorflow_serving/core:servable_state_monitor",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source_proto",
        "//tensorflow_serving/util:event_bus",
        "//tensorflow_serving/util:unique_ptr_with_deps",
        "@org_tensorflow//tensorflow/core:lib",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "server_core_test",
    srcs = ["server_core_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export.meta",
    ],
    deps = [
        ":server_core",
        "//tensorflow_serving/config:model_server_config_proto",
        "//tensorflow_serving/core:servable_state",
        "//tensorflow_serving/core/test_util:availability_test_util",
        "//tensorflow_serving/core/test_util:fake_loader_source_adapter",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers/test_util:server_core_test_util",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

SUPPORTED_TENSORFLOW_OPS = [
    "@org_tensorflow//tensorflow/contrib:contrib_kernels",
    "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
]

TENSORFLOW_DEPS = [
    "@org_tensorflow//tensorflow/core:tensorflow",
    "//tensorflow_serving/servables/tensorflow:session_bundle_source_adapter",
    "//tensorflow_serving/servables/tensorflow:predict_impl",
]

cc_binary(
    name = "tensorflow_model_server",
    srcs = [
        "main.cc",
    ],
    visibility = ["//tensorflow_serving:internal"],
    deps = [
        ":server_core",
        "@protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/platform/cloud:gcs_file_system",
        "//tensorflow_serving/apis:prediction_service_proto",
        "//tensorflow_serving/core:servable_state_monitor",
        "@grpc//:grpc++",
    ] + TENSORFLOW_DEPS + SUPPORTED_TENSORFLOW_OPS,
)

py_test(
    name = "tensorflow_model_server_test",
    size = "large",
    srcs = ["tensorflow_model_server_test.py"],
    data = [
        ":tensorflow_model_server",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_half_plus_two/00000123/export",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_half_plus_two/00000123/export.meta",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two/00000123/export.meta",
    ],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "tensorflow_model_server_test_client",
    srcs = ["tensorflow_model_server_test_client.py"],
    deps = [
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
