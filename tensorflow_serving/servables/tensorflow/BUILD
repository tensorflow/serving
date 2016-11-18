# Description: TensorFlow servables.

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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

serving_proto_library(
    name = "session_bundle_config_proto",
    srcs = ["session_bundle_config.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "bundle_factory_util",
    srcs = ["bundle_factory_util.cc"],
    hdrs = ["bundle_factory_util.h"],
    deps = [
        ":serving_session",
        ":session_bundle_config_proto",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/batching:shared_batch_scheduler",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_proto",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_test(
    name = "bundle_factory_util_test",
    size = "medium",
    srcs = ["bundle_factory_util_test.cc"],
    data = ["@org_tensorflow//tensorflow/contrib/session_bundle/example:half_plus_two"],
    deps = [
        ":bundle_factory_test_util",
        ":bundle_factory_util",
        ":session_bundle_config_proto",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/batching:shared_batch_scheduler",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resources_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "bundle_factory_test_util",
    testonly = 1,
    srcs = ["bundle_factory_test_util.cc"],
    hdrs = ["bundle_factory_test_util.h"],
    deps = [
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_proto",
        "//tensorflow_serving/test_util",
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
        ":session_bundle_config_proto",
        "//tensorflow_serving/resources:resources_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "session_bundle_factory",
    srcs = ["session_bundle_factory.cc"],
    hdrs = ["session_bundle_factory.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":bundle_factory_util",
        ":session_bundle_config_proto",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/batching:shared_batch_scheduler",
        "//tensorflow_serving/resources:resources_proto",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "session_bundle_factory_test",
    size = "medium",
    srcs = ["session_bundle_factory_test.cc"],
    data = ["@org_tensorflow//tensorflow/contrib/session_bundle/example:half_plus_two"],
    deps = [
        ":bundle_factory_test",
        ":session_bundle_config_proto",
        ":session_bundle_factory",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
        "@protobuf//:cc_wkt_protos",
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
        ":session_bundle_config_proto",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/batching:shared_batch_scheduler",
        "//tensorflow_serving/resources:resources_proto",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/contrib/session_bundle:bundle_shim",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "saved_model_bundle_factory_test",
    size = "medium",
    srcs = ["saved_model_bundle_factory_test.cc"],
    data = [
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
        "@org_tensorflow//tensorflow/contrib/session_bundle/example:half_plus_two",
    ],
    deps = [
        ":bundle_factory_test",
        ":saved_model_bundle_factory",
        ":session_bundle_config_proto",
        "//tensorflow_serving/core/test_util:test_main",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "session_bundle_source_adapter",
    srcs = ["session_bundle_source_adapter.cc"],
    hdrs = ["session_bundle_source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":serving_session",
        ":session_bundle_factory",
        ":session_bundle_source_adapter_proto",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/util:optional",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "session_bundle_source_adapter_test",
    size = "medium",
    srcs = ["session_bundle_source_adapter_test.cc"],
    data = ["@org_tensorflow//tensorflow/contrib/session_bundle/example:half_plus_two"],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":bundle_factory_test_util",
        ":session_bundle_config_proto",
        ":session_bundle_source_adapter",
        ":session_bundle_source_adapter_proto",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core/test_util:source_adapter_test_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resources_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
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
        ":saved_model_bundle_factory",
        ":session_bundle_source_adapter_proto",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/resources:resources_proto",
        "//tensorflow_serving/util:optional",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "saved_model_bundle_source_adapter_test",
    size = "medium",
    srcs = ["saved_model_bundle_source_adapter_test.cc"],
    data = [
        "@org_tensorflow//tensorflow/cc/saved_model:saved_model_half_plus_two",
        "@org_tensorflow//tensorflow/contrib/session_bundle/example:half_plus_two",
    ],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":bundle_factory_test_util",
        ":saved_model_bundle_source_adapter",
        ":session_bundle_config_proto",
        ":session_bundle_source_adapter_proto",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core/test_util:source_adapter_test_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resources_proto",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

serving_proto_library(
    name = "session_bundle_source_adapter_proto",
    srcs = ["session_bundle_source_adapter.proto"],
    cc_api_version = 2,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":session_bundle_config_proto",
    ],
)

cc_library(
    name = "simple_servers",
    srcs = ["simple_servers.cc"],
    hdrs = ["simple_servers.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":session_bundle_source_adapter",
        ":session_bundle_source_adapter_proto",
        "//tensorflow_serving/core:aspired_versions_manager_builder",
        "//tensorflow_serving/core:eager_load_policy",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/core:target",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source_proto",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "simple_servers_test",
    srcs = ["simple_servers_test.cc"],
    data = ["@org_tensorflow//tensorflow/contrib/session_bundle/example:half_plus_two"],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":simple_servers",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util:unique_ptr_with_deps",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "serving_session",
    srcs = ["serving_session.cc"],
    hdrs = ["serving_session.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "predict_impl",
    srcs = ["predict_impl.cc"],
    hdrs = ["predict_impl.h"],
    deps = [
        "//tensorflow_serving/apis:predict_proto",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/model_servers:server_core",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
