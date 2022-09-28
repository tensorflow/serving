# Description: TensorFlow servables.

load("//tensorflow_serving:oss_or_google.bzl", "if_google", "if_oss")
load("//tensorflow_serving:serving.bzl", "serving_proto_library", "serving_proto_library_py", "serving_tensorflow_proto_dep")

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

serving_proto_library(
    name = "session_bundle_config_proto",
    srcs = ["session_bundle_config.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_protobuf//:cc_wkt_protos",
        serving_tensorflow_proto_dep(
            "@org_tensorflow//tensorflow/core:protos_all",
        ),
    ],
)

serving_proto_library_py(
    name = "session_bundle_config_py_pb2",
    proto_library = "session_bundle_config_proto",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "resource_estimator",
    hdrs = ["resource_estimator.h"],
    deps = ["//tensorflow_serving/util:oss_or_google"] + if_oss([
        "//tensorflow_serving/servables/tensorflow/oss:resource_estimator",
    ]) + if_google(["//tensorflow_serving/servables/tensorflow/google:resource_estimator"]),
)

cc_library(
    name = "bundle_factory_util",
    srcs = ["bundle_factory_util.cc"],
    hdrs = ["bundle_factory_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":resource_estimator",
        ":serving_session",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_cc_proto",
        "//tensorflow_serving/util:file_probing_env",
        "//tensorflow_serving/util:proto_util",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:batch_scheduler",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:shared_batch_scheduler",
    ],
)

cc_test(
    name = "bundle_factory_util_test",
    size = "medium",
    srcs = ["bundle_factory_util_test.cc"],
    data = [
        "//tensorflow_serving/session_bundle:session_bundle_half_plus_two",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":bundle_factory_test_util",
        ":bundle_factory_util",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resources_cc_proto",
        "//tensorflow_serving/session_bundle:session_bundle_util",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util/test_util:mock_file_probing_env",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:shared_batch_scheduler",
    ],
)

cc_library(
    name = "bundle_factory_test_util",
    testonly = 1,
    srcs = ["bundle_factory_test_util.cc"],
    hdrs = ["bundle_factory_test_util.h"],
    deps = [
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_cc_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "bundle_factory_test",
    testonly = 1,
    srcs = ["bundle_factory_test.h"],
    deps = [
        ":bundle_factory_test_util",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/resources:resources_cc_proto",
        "//tensorflow_serving/test_util",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "saved_model_bundle_factory",
    srcs = ["saved_model_bundle_factory.cc"],
    hdrs = ["saved_model_bundle_factory.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":bundle_factory_util",
        ":session_bundle_config_cc_proto",
        ":tflite_session_lib",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/resources:resources_cc_proto",
        "//tensorflow_serving/session_bundle:session_bundle_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:shared_batch_scheduler",
    ],
)

cc_test(
    name = "saved_model_bundle_factory_test",
    size = "medium",
    srcs = ["saved_model_bundle_factory_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tflite",
        "//tensorflow_serving/session_bundle:session_bundle_half_plus_two",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":bundle_factory_test",
        ":bundle_factory_test_util",
        ":saved_model_bundle_factory",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/core/test_util:session_test_util",
        "//tensorflow_serving/core/test_util:test_main",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "saved_model_bundle_source_adapter",
    srcs = ["saved_model_bundle_source_adapter.cc"],
    hdrs = ["saved_model_bundle_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":machine_learning_metadata",
        ":saved_model_bundle_factory",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":saved_model_warmup",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/resources:resource_util",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_cc_proto",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

cc_test(
    name = "saved_model_bundle_source_adapter_test",
    size = "medium",
    srcs = ["saved_model_bundle_source_adapter_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_mlmd",
        "//tensorflow_serving/session_bundle:session_bundle_half_plus_two",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":bundle_factory_test_util",
        ":saved_model_bundle_source_adapter",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core/test_util:session_test_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resources_cc_proto",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util:oss_or_google",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

serving_proto_library(
    name = "saved_model_bundle_source_adapter_proto",
    srcs = ["saved_model_bundle_source_adapter.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":session_bundle_config_proto",
    ],
)

serving_proto_library_py(
    name = "saved_model_bundle_source_adapter_py_pb2",
    proto_library = "saved_model_bundle_source_adapter_proto",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "simple_servers",
    srcs = ["simple_servers.cc"],
    hdrs = ["simple_servers.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":saved_model_bundle_source_adapter",
        ":saved_model_bundle_source_adapter_cc_proto",
        "//tensorflow_serving/config:file_system_storage_path_source_cc_proto",
        "//tensorflow_serving/core:aspired_versions_manager_builder",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/core:target",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "simple_servers_test",
    srcs = ["simple_servers_test.cc"],
    data = ["@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two"],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":simple_servers",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "serving_session",
    srcs = ["serving_session.cc"],
    hdrs = ["serving_session.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tflite_session_lib",
    srcs = [
        "tflite_interpreter_pool.cc",
        "tflite_session.cc",
    ],
    hdrs = [
        "tflite_interpreter_pool.h",
        "tflite_session.h",
    ],
    deps = [
        ":serving_session",
        "//tensorflow_serving/batching:incremental_barrier",
        "//tensorflow_serving/batching:threadsafe_status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:session_options",
        "@org_tensorflow//tensorflow/core/kernels/batching_util:basic_batch_scheduler",
        "@org_tensorflow//tensorflow/lite:external_cpu_backend_context",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite:util",
        "@org_tensorflow//tensorflow/lite/c:c_api",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/delegates/flex:delegate",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor_utils",
        "@org_tensorflow//tensorflow/lite/kernels/parse_example",
        "@org_tensorflow//tensorflow/lite/profiling:profile_summarizer",
        "@org_tensorflow//tensorflow/lite/profiling:profile_summary_formatter",
        "@org_tensorflow//tensorflow/lite/profiling:profiler",
        "@org_tensorflow//tensorflow/lite/tools/signature:signature_def_util",
    ],
)

cc_test(
    name = "tflite_session_test",
    size = "small",
    srcs = ["tflite_session_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:mobilenet_v1_quant_tflite",
        "//tensorflow_serving/servables/tensorflow/testdata:parse_example_tflite",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tflite",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tflite_with_sigdef",
    ],
    deps = [
        ":tflite_session_lib",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/functional:bind_front",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite:util",
        "@org_tensorflow//tensorflow/lite:version",
        "@org_tensorflow//tensorflow/lite/tools/signature:signature_def_util",
    ],
)

cc_test(
    name = "tflite_interpreter_pool_test",
    size = "small",
    srcs = ["tflite_interpreter_pool_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:mobilenet_v1_quant_tflite",
        "//tensorflow_serving/servables/tensorflow/testdata:parse_example_tflite",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tflite",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tflite_with_sigdef",
    ],
    deps = [
        ":tflite_session_lib",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_protobuf//:protobuf",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite:util",
        "@org_tensorflow//tensorflow/lite:version",
        "@org_tensorflow//tensorflow/lite/tools/signature:signature_def_util",
    ],
)

cc_binary(
    name = "tflite_session",
    srcs = ["tflite_session_main.cc"],
    deps = [
        ":tflite_session_lib",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "predict_impl",
    srcs = ["predict_impl.cc"],
    hdrs = ["predict_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":predict_util",
        ":thread_pool_factory",
        ":util",
        "//tensorflow_serving/apis:predict_cc_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:server_core",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "predict_impl_test",
    size = "medium",
    srcs = ["predict_impl_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_counter/00000123/saved_model.pb",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_counter/00000123/variables/variables.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_counter/00000123/variables/variables.index",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":predict_impl",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/servables/tensorflow/test_util:fake_thread_pool_factory",
        "//tensorflow_serving/servables/tensorflow/test_util:fake_thread_pool_factory_cc_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "predict_response_tensor_serialization_option",
    hdrs = ["predict_response_tensor_serialization_option.h"],
)

cc_library(
    name = "predict_util",
    srcs = ["predict_util.cc"],
    hdrs = ["predict_util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":predict_response_tensor_serialization_option",
        ":util",
        "//tensorflow_serving/apis:predict_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "predict_util_test",
    size = "medium",
    srcs = ["predict_util_test.cc"],
    data = [
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_half_plus_two/00000123/export",
        "//tensorflow_serving/servables/tensorflow/testdata:bad_half_plus_two/00000123/export.meta",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_counter/00000123/saved_model.pb",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_counter/00000123/variables/variables.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_counter/00000123/variables/variables.index",
    ] + if_google([
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.index",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.meta",
    ]),
    deps = [
        ":predict_util",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":session_bundle_config_cc_proto",
        ":util",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util:oss_or_google",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "get_model_metadata_impl",
    srcs = ["get_model_metadata_impl.cc"],
    hdrs = ["get_model_metadata_impl.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/apis:get_model_metadata_cc_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:server_core",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "get_model_metadata_impl_test",
    size = "medium",
    srcs = ["get_model_metadata_impl_test.cc"],
    data = [
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ] + if_google([
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.data-00000-of-00001",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.index",
        "//tensorflow_serving/servables/tensorflow/google/testdata:half_plus_two/00000123/export.meta",
    ]),
    deps = [
        ":get_model_metadata_impl",
        ":saved_model_bundle_source_adapter_cc_proto",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/config:model_server_config_cc_proto",
        "//tensorflow_serving/config:platform_config_cc_proto",
        "//tensorflow_serving/core:aspired_version_policy",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util:oss_or_google",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "classifier",
    srcs = ["classifier.cc"],
    hdrs = ["classifier.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:classifier",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/cc/saved_model:loader_lite",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "classifier_test",
    size = "medium",
    srcs = ["classifier_test.cc"],
    deps = [
        ":classifier",
        ":util",
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/core/test_util:mock_session",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "classification_service",
    srcs = ["classification_service.cc"],
    hdrs = ["classification_service.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":classifier",
        ":util",
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:classifier",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:server_core",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "classification_service_test",
    srcs = ["classification_service_test.cc"],
    data = [
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":classification_service",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/config:model_server_config_cc_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "regression_service",
    srcs = ["regression_service.cc"],
    hdrs = ["regression_service.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":regressor",
        "//tensorflow_serving/apis:regression_cc_proto",
        "//tensorflow_serving/apis:regressor",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/servables/tensorflow:util",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "regression_service_test",
    srcs = ["regression_service_test.cc"],
    data = [
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":regression_service",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/config:model_server_config_cc_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "regressor",
    srcs = ["regressor.cc"],
    hdrs = ["regressor.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":util",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:regression_cc_proto",
        "//tensorflow_serving/apis:regressor",
        "@com_google_absl//absl/types:optional",
        "@org_tensorflow//tensorflow/cc/saved_model:loader_lite",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "regressor_test",
    size = "medium",
    srcs = ["regressor_test.cc"],
    deps = [
        ":regressor",
        ":util",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:regression_cc_proto",
        "//tensorflow_serving/core/test_util:mock_session",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "multi_inference",
    srcs = ["multi_inference.cc"],
    hdrs = ["multi_inference.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/apis:inference_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/servables/tensorflow:classifier",
        "//tensorflow_serving/servables/tensorflow:regressor",
        "//tensorflow_serving/servables/tensorflow:util",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "multi_inference_test",
    size = "medium",
    srcs = ["multi_inference_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tf2_cpu",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":multi_inference",
        ":util",
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:regression_cc_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_cc_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "multi_inference_helper",
    srcs = ["multi_inference_helper.cc"],
    hdrs = ["multi_inference_helper.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":multi_inference",
        "//tensorflow_serving/apis:inference_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/model_servers:server_core",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "multi_inference_helper_test",
    size = "medium",
    srcs = ["multi_inference_helper_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_tf2_cpu",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":multi_inference_helper",
        ":util",
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:regression_cc_proto",
        "//tensorflow_serving/core:availability_preserving_policy",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/model_servers:model_platform_types",
        "//tensorflow_serving/model_servers:platform_config_util",
        "//tensorflow_serving/model_servers:server_core",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_cc_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis/internal:serialized_input_cc_proto",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_cc_proto",
        "//tensorflow_serving/util:file_probing_env",
        "//tensorflow_serving/util:threadpool_executor",
        "@com_google_absl//absl/types:optional",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/servables/tensorflow:bundle_factory_test_util",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util/test_util:mock_file_probing_env",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "saved_model_warmup_util",
    srcs = ["saved_model_warmup_util.cc"],
    hdrs = ["saved_model_warmup_util.h"],
    deps = [
        ":session_bundle_config_cc_proto",
        ":util",
        "//tensorflow_serving/apis:prediction_log_cc_proto",
        "//tensorflow_serving/util:threadpool_executor",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "saved_model_warmup_test_util",
    testonly = 1,
    srcs = ["saved_model_warmup_test_util.cc"],
    hdrs = ["saved_model_warmup_test_util.h"],
    deps = [
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:inference_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:predict_cc_proto",
        "//tensorflow_serving/apis:prediction_log_cc_proto",
        "//tensorflow_serving/apis:regression_cc_proto",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "saved_model_warmup_util_test",
    size = "small",
    srcs = ["saved_model_warmup_util_test.cc"],
    deps = [
        ":saved_model_warmup_test_util",
        ":saved_model_warmup_util",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/apis:classification_cc_proto",
        "//tensorflow_serving/apis:inference_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:predict_cc_proto",
        "//tensorflow_serving/apis:prediction_log_cc_proto",
        "//tensorflow_serving/apis:regression_cc_proto",
        "//tensorflow_serving/core/test_util:test_main",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/cc/saved_model:signature_constants",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "saved_model_warmup",
    srcs = ["saved_model_warmup.cc"],
    hdrs = ["saved_model_warmup.h"],
    deps = [
        ":classifier",
        ":multi_inference",
        ":predict_util",
        ":regressor",
        ":saved_model_warmup_util",
        ":session_bundle_config_cc_proto",
        ":util",
        "//tensorflow_serving/apis:prediction_log_cc_proto",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_test(
    name = "saved_model_warmup_test",
    size = "small",
    srcs = ["saved_model_warmup_test.cc"],
    deps = [
        ":saved_model_warmup",
        ":saved_model_warmup_test_util",
        ":session_bundle_config_cc_proto",
        "//tensorflow_serving/apis:input_cc_proto",
        "//tensorflow_serving/apis:model_cc_proto",
        "//tensorflow_serving/apis:prediction_log_cc_proto",
        "//tensorflow_serving/core/test_util:mock_session",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "machine_learning_metadata",
    srcs = ["machine_learning_metadata.cc"],
    hdrs = ["machine_learning_metadata.h"],
    deps = [
        "@org_tensorflow//tensorflow/cc/saved_model:constants",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "machine_learning_metadata_test",
    size = "small",
    srcs = ["machine_learning_metadata_test.cc"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_mlmd",
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        ":bundle_factory_test_util",
        ":machine_learning_metadata",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

serving_proto_library(
    name = "thread_pool_factory_config_proto",
    srcs = ["thread_pool_factory_config.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "thread_pool_factory",
    srcs = ["thread_pool_factory.cc"],
    hdrs = ["thread_pool_factory.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow_serving/util:class_registration",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "thread_pool_factory_test",
    size = "small",
    srcs = ["thread_pool_factory_test.cc"],
    deps = [
        ":thread_pool_factory",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/platform:threadpool_options",
    ],
)
