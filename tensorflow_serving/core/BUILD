# Description: Tensorflow Serving core.

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
    name = "servable_id",
    hdrs = ["servable_id.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "servable_id_test",
    srcs = ["servable_id_test.cc"],
    deps = [
        ":servable_id",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "servable_data",
    hdrs = ["servable_data.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        ":servable_id",
    ],
)

cc_test(
    name = "servable_data_test",
    srcs = ["servable_data_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":servable_data",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "loader",
    hdrs = ["loader.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        ":source",
        "//tensorflow_serving/resources:resources_proto",
        "//tensorflow_serving/util:any_ptr",
    ],
)

cc_library(
    name = "simple_loader",
    hdrs = ["simple_loader.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        ":loader",
        ":source_adapter",
        "//tensorflow_serving/util:any_ptr",
        "//tensorflow_serving/util:optional",
    ],
)

cc_test(
    name = "simple_loader_test",
    srcs = [
        "simple_loader_test.cc",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:test",
        ":servable_data",
        ":simple_loader",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
    ],
)

cc_library(
    name = "source",
    hdrs = ["source.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":servable_data",
    ],
)

cc_library(
    name = "target",
    hdrs = ["target.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":servable_data",
        ":source",
        "//tensorflow_serving/util:observer",
    ],
)

cc_library(
    name = "source_adapter",
    hdrs = ["source_adapter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":servable_data",
        ":source",
        ":target",
    ],
)

cc_test(
    name = "source_adapter_test",
    srcs = ["source_adapter_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":servable_id",
        ":source_adapter",
        ":storage_path",
        "//tensorflow_serving/core/test_util:mock_storage_path_target",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "source_router",
    hdrs = ["source_router.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":source",
        ":source_adapter",
        ":target",
    ],
)

cc_test(
    name = "source_router_test",
    srcs = ["source_router_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":servable_data",
        ":source_router",
        ":storage_path",
        "//tensorflow_serving/core/test_util:mock_storage_path_target",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "static_source_router",
    hdrs = ["static_source_router.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        ":source_router",
    ],
)

cc_test(
    name = "static_source_router_test",
    srcs = ["static_source_router_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":servable_data",
        ":source",
        ":static_source_router",
        ":storage_path",
        ":target",
        "//tensorflow_serving/core/test_util:mock_storage_path_target",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "storage_path",
    hdrs = ["storage_path.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":servable_data",
        ":servable_id",
    ],
)

cc_test(
    name = "storage_path_test",
    srcs = ["storage_path_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":storage_path",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "servable_state",
    hdrs = ["servable_state.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":servable_id",
    ],
)

cc_library(
    name = "servable_handle",
    hdrs = ["servable_handle.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:tensorflow",
        ":loader",
        "//tensorflow_serving/util:any_ptr",
    ],
)

cc_library(
    name = "manager",
    hdrs = ["manager.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":servable_handle",
        ":servable_id",
        "//tensorflow_serving/util:optional",
    ],
)

cc_test(
    name = "manager_test",
    srcs = ["manager_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":manager",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util:any_ptr",
    ],
)

cc_library(
    name = "manager_wrapper",
    srcs = ["manager_wrapper.cc"],
    hdrs = ["manager_wrapper.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":manager",
        ":servable_id",
        "//tensorflow_serving/util:unique_ptr_with_deps",
    ],
)

cc_library(
    name = "aspired_versions_manager_builder",
    srcs = ["aspired_versions_manager_builder.cc"],
    hdrs = ["aspired_versions_manager_builder.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":aspired_versions_manager",
        ":loader",
        ":manager_wrapper",
        ":source",
        ":source_adapter",
        "//tensorflow_serving/util:unique_ptr_with_deps",
    ],
)

cc_test(
    name = "aspired_versions_manager_builder_test",
    srcs = ["aspired_versions_manager_builder_test.cc"],
    deps = [
        ":aspired_versions_manager_builder",
        ":eager_load_policy",
        ":servable_data",
        ":servable_handle",
        ":servable_state_monitor",
        ":simple_loader",
        ":storage_path",
        "//tensorflow_serving/core/test_util:availability_test_util",
        "//tensorflow_serving/core/test_util:source_adapter_test_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util:event_bus",
    ],
)

cc_library(
    name = "availability_helpers",
    srcs = ["availability_helpers.cc"],
    hdrs = ["availability_helpers.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":manager",
        ":servable_id",
    ],
)

cc_test(
    name = "availability_helpers_test",
    srcs = ["availability_helpers_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":availability_helpers",
        ":servable_handle",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "servable_state_monitor",
    srcs = ["servable_state_monitor.cc"],
    hdrs = ["servable_state_monitor.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":servable_id",
        ":servable_state",
        "//tensorflow_serving/util:event_bus",
        "//tensorflow_serving/util:optional",
    ],
)

cc_test(
    name = "servable_state_monitor_test",
    srcs = ["servable_state_monitor_test.cc"],
    deps = [
        ":servable_state_monitor",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
    ],
)

cc_library(
    name = "loader_harness",
    srcs = ["loader_harness.cc"],
    hdrs = ["loader_harness.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        ":loader",
        ":servable_id",
        "//tensorflow_serving/util:optional",
    ],
)

cc_test(
    name = "loader_harness_test",
    srcs = ["loader_harness_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:tensorflow",
        "@tf//tensorflow/core:test",
        ":loader_harness",
        "//tensorflow_serving/core/test_util:mock_loader",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util:any_ptr",
    ],
)

cc_library(
    name = "basic_manager",
    srcs = ["basic_manager.cc"],
    hdrs = ["basic_manager.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":loader",
        ":loader_harness",
        ":manager",
        ":servable_data",
        ":servable_handle",
        ":servable_id",
        ":servable_state",
        "//tensorflow_serving/resources:resource_tracker",
        "//tensorflow_serving/util:cleanup",
        "//tensorflow_serving/util:event_bus",
        "//tensorflow_serving/util:executor",
        "//tensorflow_serving/util:fast_read_dynamic_ptr",
        "//tensorflow_serving/util:inline_executor",
        "//tensorflow_serving/util:threadpool_executor",
    ],
)

cc_test(
    name = "basic_manager_test",
    size = "small",
    srcs = ["basic_manager_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:test",
        ":availability_helpers",
        ":basic_manager",
        ":servable_state_monitor",
        "//tensorflow_serving/core/test_util:availability_test_util",
        "//tensorflow_serving/core/test_util:fake_loader",
        "//tensorflow_serving/core/test_util:mock_loader",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util:any_ptr",
    ],
)

cc_library(
    name = "aspired_versions_manager",
    srcs = ["aspired_versions_manager.cc"],
    hdrs = ["aspired_versions_manager.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":aspired_version_policy",
        ":basic_manager",
        ":loader",
        ":manager",
        ":servable_data",
        ":servable_handle",
        ":servable_id",
        ":servable_state",
        ":source",
        ":target",
        "//tensorflow_serving/util:event_bus",
        "//tensorflow_serving/util:optional",
        "//tensorflow_serving/util:periodic_function",
    ],
)

cc_test(
    name = "aspired_versions_manager_test",
    size = "small",
    srcs = ["aspired_versions_manager_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:test",
        ":aspired_versions_manager",
        ":availability_helpers",
        ":eager_load_policy",
        ":servable_state_monitor",
        "//tensorflow_serving/core/test_util:availability_test_util",
        "//tensorflow_serving/core/test_util:fake_loader",
        "//tensorflow_serving/core/test_util:manager_test_util",
        "//tensorflow_serving/core/test_util:mock_loader",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util:any_ptr",
    ],
)

cc_test(
    name = "aspired_versions_manager_benchmark",
    srcs = ["aspired_versions_manager_benchmark.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:tensorflow",
        "@tf//tensorflow/core:test",
        ":aspired_version_policy",
        ":aspired_versions_manager",
        ":eager_load_policy",
        ":loader",
        ":manager",
        ":servable_data",
        ":servable_handle",
        ":simple_loader",
        "//tensorflow_serving/core/test_util:manager_test_util",
        "//tensorflow_serving/util:periodic_function",
    ],
)

cc_library(
    name = "static_manager",
    srcs = ["static_manager.cc"],
    hdrs = ["static_manager.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":basic_manager",
        ":manager",
        ":servable_handle",
        ":servable_id",
        ":simple_loader",
    ],
)

cc_test(
    name = "static_manager_test",
    srcs = ["static_manager_test.cc"],
    deps = [
        ":static_manager",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util:fake_clock_env",
    ],
)

cc_library(
    name = "caching_manager",
    srcs = ["caching_manager.cc"],
    hdrs = ["caching_manager.h"],
    deps = [
        "@tf//tensorflow/core:lib",
        ":basic_manager",
        ":loader",
        ":manager",
        ":servable_data",
        ":servable_handle",
        ":servable_id",
        "//tensorflow_serving/util:optional",
    ],
)

cc_test(
    name = "caching_manager_test",
    srcs = ["caching_manager_test.cc"],
    deps = [
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:test",
        ":caching_manager",
        ":servable_data",
        ":servable_handle",
        ":servable_id",
        ":servable_state",
        ":servable_state_monitor",
        ":simple_loader",
        "//tensorflow_serving/core/test_util:manager_test_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/util:event_bus",
        "//tensorflow_serving/util:optional",
        "//tensorflow_serving/util:threadpool_executor",
    ],
)

cc_library(
    name = "aspired_version_policy",
    hdrs = ["aspired_version_policy.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@tf//tensorflow/core:lib",
        ":loader_harness",
        ":servable_id",
        "//tensorflow_serving/util:optional",
    ],
)

cc_library(
    name = "eager_load_policy",
    srcs = ["eager_load_policy.cc"],
    hdrs = ["eager_load_policy.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":aspired_version_policy",
        ":loader_harness",
        "//tensorflow_serving/util:optional",
    ],
)

cc_test(
    name = "eager_load_policy_test",
    srcs = ["eager_load_policy_test.cc"],
    deps = [
        ":eager_load_policy",
        ":servable_id",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)

cc_library(
    name = "eager_unload_policy",
    srcs = ["eager_unload_policy.cc"],
    hdrs = ["eager_unload_policy.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":aspired_version_policy",
        ":loader_harness",
        "//tensorflow_serving/util:optional",
    ],
)

cc_test(
    name = "eager_unload_policy_test",
    srcs = ["eager_unload_policy_test.cc"],
    deps = [
        ":eager_unload_policy",
        ":servable_id",
        "//tensorflow_serving/core/test_util:test_main",
    ],
)
