# Description: Model Server

load("//tensorflow_serving:oss_or_google.bzl", "if_google")
load("//tensorflow_serving:tensorflow_version.bzl", "if_not_v2", "if_v2")
load("@rules_pkg//:pkg.bzl", "pkg_deb", "pkg_tar")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_libtpu")

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])

# Visibility group for clients that link in custom ops, while using the
# vanilla TensorFlow ModelServer without any other modifications. Exclusively
# for adding custom TensorFlow ops. Does not grant access to private headers
# or other implementation details.
package_group(name = "tensorflow_model_server_custom_op_clients")

package_group(
    name = "testing",
    packages = [
        # tensorflow_transform for testing only
        "//third_party/py/tensorflow_transform/examples/...",
        "//third_party/py/tensorflow_transform/google/examples/...",
    ],
)

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
    name = "model_platform_types",
    hdrs = ["model_platform_types.h"],
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "platform_config_util",
    srcs = ["platform_config_util.cc"],
    hdrs = ["platform_config_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":model_platform_types",
        "//tensorflow_serving/config:platform_config_cc_proto",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter_cc_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_cc_proto",
    ],
)

cc_library(
    name = "server_core",
    srcs = ["server_core.cc"],
    hdrs = ["server_core.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":model_platform_types",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/config:file_system_storage_path_source_cc_proto",
        "//tensorflow_serving/config:logging_config_cc_proto",
        "//tensorflow_serving/config:model_server_config_cc_proto",
        "//tensorflow_serving/config:platform_config_cc_proto",
        "//tensorflow_serving/core:aspired_versions_manager",
        "//tensorflow_serving/core:dynamic_source_router",
        "//tensorflow_serving/core:load_servables_fast",
        "//tensorflow_serving/core:prefix_storage_path_source_adapter",
        "//tensorflow_serving/core:servable_state_monitor",
        "//tensorflow_serving/core:server_request_logger",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/servables/tensorflow:predict_util",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
        "//tensorflow_serving/util:event_bus",
        "//tensorflow_serving/util:unique_ptr_with_deps",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "server_core_test",
    size = "medium",
    srcs = ["server_core_test.cc"],
    deps = [
        ":model_platform_types",
        ":server_core",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:predict_cc_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_state",
        "//tensorflow_serving/core/test_util:availability_test_util",
        "//tensorflow_serving/core/test_util:fake_loader_source_adapter_cc_proto",
        "//tensorflow_serving/core/test_util:fake_log_collector",
        "//tensorflow_serving/core/test_util:mock_request_logger",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers/test_util:server_core_test_util",
        "//tensorflow_serving/model_servers/test_util:storage_path_error_injecting_source_adapter",
        "//tensorflow_serving/model_servers/test_util:storage_path_error_injecting_source_adapter_cc_proto",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util:oss_or_google",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "get_model_status_impl",
    srcs = ["get_model_status_impl.cc"],
    hdrs = ["get_model_status_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":server_core",
        "//tensorflow_serving/apis:get_model_status_cc_proto",
        "//tensorflow_serving/apis:status_cc_proto",
        "//tensorflow_serving/core:servable_state",
        "//tensorflow_serving/core:servable_state_monitor",
        "//tensorflow_serving/util:status_util",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "get_model_status_impl_test",
    size = "medium",
    srcs = ["get_model_status_impl_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_2_versions",
    ],
    deps = [
        ":get_model_status_impl",
        ":model_platform_types",
        ":platform_config_util",
        ":server_core",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter_cc_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_cc_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_test(
    name = "http_rest_api_handler_test",
    size = "medium",
    srcs = ["http_rest_api_handler_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_2_versions_metadata.json",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":http_rest_api_handler",
        ":model_platform_types",
        ":platform_config_util",
        ":server_core",
        ":server_init",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter_cc_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_cc_proto",
        "//tensorflow_serving/test_util",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_googlesource_code_re2//:re2",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "model_service_impl",
    srcs = ["model_service_impl.cc"],
    hdrs = ["model_service_impl.h"],
    deps = [
        ":get_model_status_impl",
        ":grpc_status_util",
        ":server_core",
        "//tensorflow_serving/apis:model_management_cc_proto",
        "//tensorflow_serving/apis:model_service_cc_proto",
        "//tensorflow_serving/util:status_util",
    ],
)

cc_library(
    name = "prediction_service_util",
    srcs = ["prediction_service_util.cc"],
    hdrs = ["prediction_service_util.h"],
    deps = [
        ":server_core",
        "//tensorflow_serving/apis:prediction_service_cc_proto",
        "//tensorflow_serving/servables/tensorflow:thread_pool_factory",
    ],
)

cc_library(
    name = "prediction_service_impl",
    srcs = ["prediction_service_impl.cc"],
    hdrs = ["prediction_service_impl.h"],
    deps = [
        ":grpc_status_util",
        ":prediction_service_util",
        "//tensorflow_serving/apis:prediction_service_cc_proto",
        "//tensorflow_serving/servables/tensorflow:classification_service",
        "//tensorflow_serving/servables/tensorflow:get_model_metadata_impl",
        "//tensorflow_serving/servables/tensorflow:multi_inference_helper",
        "//tensorflow_serving/servables/tensorflow:predict_impl",
        "//tensorflow_serving/servables/tensorflow:regression_service",
        "//tensorflow_serving/servables/tensorflow:thread_pool_factory",
        "//tensorflow_serving/servables/tensorflow:util",
        "@com_github_grpc_grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "grpc_status_util",
    srcs = ["grpc_status_util.cc"],
    hdrs = ["grpc_status_util.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "http_server",
    srcs = ["http_server.cc"],
    hdrs = ["http_server.h"],
    deps = [
        ":http_rest_api_handler",
        ":http_rest_api_util",
        ":server_core",
        ":server_init",
        "//tensorflow_serving/config:monitoring_config_cc_proto",
        "//tensorflow_serving/servables/tensorflow:util",
        "//tensorflow_serving/util:prometheus_exporter",
        "//tensorflow_serving/util:threadpool_executor",
        "//tensorflow_serving/util/net_http/server/public:http_server",
        "//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_googlesource_code_re2//:re2",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "http_rest_api_util",
    srcs = ["http_rest_api_util.cc"],
    hdrs = ["http_rest_api_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_serving/apis:get_model_metadata_cc_proto",
        "//tensorflow_serving/apis:get_model_status_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/servables/tensorflow:get_model_metadata_impl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_googlesource_code_re2//:re2",
        "@curl",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "http_rest_api_util_test",
    srcs = ["http_rest_api_util_test.cc"],
    deps = [
        ":http_rest_api_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "http_rest_api_handler_base",
    hdrs = ["http_rest_api_handler_base.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "http_rest_api_handler",
    srcs = ["http_rest_api_handler.cc"],
    hdrs = ["http_rest_api_handler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":get_model_status_impl",
        ":http_rest_api_handler_base",
        ":http_rest_api_util",
        ":server_core",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:predict_cc_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/servables/tensorflow:classification_service",
        "//tensorflow_serving/servables/tensorflow:get_model_metadata_impl",
        "//tensorflow_serving/servables/tensorflow:predict_impl",
        "//tensorflow_serving/servables/tensorflow:regression_service",
        "//tensorflow_serving/util:json_tensor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_googlesource_code_re2//:re2",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

# This is not inlined by tools bcos inlining non-constant values
# is not provably safe.
SUPPORTED_TENSORFLOW_OPS = if_v2([]) + if_not_v2([
    "@org_tensorflow//tensorflow/contrib:contrib_kernels",
    "@org_tensorflow//tensorflow/contrib:contrib_ops_op_lib",
]) + [
    "@org_tensorflow_text//tensorflow_text:ops_lib",
    # Support for TensorFlow Decision Forests inference ops.
    # [September 2022] This dependency adds 3 MB to the model server (352 MB -> 355 MB i.e. +0.8%).
    "@org_tensorflow_decision_forests//tensorflow_decision_forests/tensorflow/ops/inference:kernel_and_op",
]

cc_library(
    name = "server_init",
    srcs = [
        "server_init.cc",
    ],
    hdrs = ["server_init.h"],
    deps = [
        ":http_rest_api_handler",
        ":http_rest_api_handler_base",
        ":platform_config_util",
        ":prediction_service_impl",
        ":server_core",
        "//tensorflow_serving/apis:prediction_service_cc_proto",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "server_lib",
    srcs = [
        "server.cc",
    ],
    hdrs = ["server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":http_server",
        ":server_init",
        ":model_platform_types",
        ":prediction_service_impl",
        ":server_core",
        ":grpc_status_util",
        ":model_service_impl",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_github_grpc_grpc//:grpc++",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/core/profiler/rpc:profiler_service_impl",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "//tensorflow_serving/config:model_server_config_cc_proto",
        "//tensorflow_serving/config:monitoring_config_cc_proto",
        "//tensorflow_serving/config:ssl_config_cc_proto",
        "//tensorflow_serving/config:platform_config_cc_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_cc_proto",
        "//tensorflow_serving/util:proto_util",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "//tensorflow_serving/servables/tensorflow:classification_service",
        "//tensorflow_serving/servables/tensorflow:get_model_metadata_impl",
        "//tensorflow_serving/servables/tensorflow:multi_inference",
        "//tensorflow_serving/servables/tensorflow:regression_service",
        "//tensorflow_serving/servables/tensorflow:saved_model_bundle_source_adapter",
        "//tensorflow_serving/servables/tensorflow:predict_impl",
        "//tensorflow_serving/servables/tensorflow:util",
        "//tensorflow_serving/servables/tensorflow:thread_pool_factory",
        "//tensorflow_serving/servables/tensorflow:thread_pool_factory_config_cc_proto",
    ] + SUPPORTED_TENSORFLOW_OPS,
)

cc_library(
    name = "tensorflow_model_server_main_lib",
    srcs = [
        "main.cc",
    ],
    hdrs = [
        "version.h",
    ],
    linkstamp = "version.cc",
    visibility = [
        ":tensorflow_model_server_custom_op_clients",
        "//tensorflow_serving:internal",
    ],
    deps = [
        ":server_lib",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/platform/cloud:gcs_file_system",
    ] + if_libtpu([
        "@org_tensorflow//tensorflow/core/tpu:tpu_global_init",
        "@org_tensorflow//tensorflow/core/tpu:tpu_api_dlsym_initializer",
    ]),
)

cc_binary(
    name = "tensorflow_model_server",
    stamp = 1,
    visibility = [
        ":testing",
        "//tensorflow_serving:internal",
    ],
    deps = [
        ":tensorflow_model_server_main_lib",
    ],
)

py_test(
    name = "tensorflow_model_server_test",
    size = "medium",
    srcs = ["tensorflow_model_server_test.py"],
    data = [
        ":tensorflow_model_server",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_half_plus_two/00000123/export",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_half_plus_two/00000123/export.meta",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_model_config.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:batching_config.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:good_model_config.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:half_plus_two_model_metadata.json",
        "//tensorflow_serving/servables/tensorflow/testdata:monitoring_config.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_three/00000123/assets/foo.txt",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_three/00000123/saved_model.pb",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_three/00000123/variables/variables.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_three/00000123/variables/variables.index",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_cpu",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tf2_cpu",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tflite",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tflite_with_sigdef",
        "//tensorflow_serving/servables/tensorflow/testdata:tf_text_regression",
    ] + if_google([
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.index",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.meta",
    ]),
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "manual",  # Missing grpc dependency.
        "noasan",  # b/136281879
        "nomsan",  # b/165917680
    ],
    deps = [
        "//tensorflow_serving/apis:model_service_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "//tensorflow_serving/model_servers/test_util:tensorflow_model_server_test_base",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "tensorflow_model_server_test_client",
    srcs = ["tensorflow_model_server_test_client.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = ["//tensorflow_serving/apis:prediction_service_proto_py_pb2"],
)

py_binary(
    name = "profiler_client",
    srcs = ["profiler_client.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
)

pkg_tar(
    name = "tensorflow_model_server_tar",
    srcs = [
        ":tensorflow_model_server",
    ],
    package_dir = "/usr/bin",
)

# Build with '-c opt --copt=-mavx --copt=-msse4.2'
pkg_deb(
    name = "tensorflow_model_server_deb",
    data = ":tensorflow_model_server_tar",
    description = "TensorFlow Serving ModelServer",
    homepage = "https://github.com/tensorflow/serving",
    maintainer = "TensorFlow Serving team",
    package = "tensorflow-model-server",
    version = "2.11.0",  # Set when releasing a new version of TensorFlow Serving (e.g. 1.0.0).
)

# Build with '-c opt'
pkg_deb(
    name = "tensorflow_model_server_universal_deb",
    data = ":tensorflow_model_server_tar",
    description = "TensorFlow Serving ModelServer",
    homepage = "https://github.com/tensorflow/serving",
    maintainer = "TensorFlow Serving team",
    package = "tensorflow-model-server-universal",
    version = "2.11.0",  # Set when releasing a new version of TensorFlow Serving (e.g. 1.0.0).
)
