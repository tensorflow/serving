# Description: Tensorflow Serving Caffe servable.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")
load("@caffe_tools//:config.bzl", "if_pycaffe", "caffe_pkg")

serving_proto_library(
    name = "caffe_session_bundle_config_proto",
    srcs = ["caffe_session_bundle_config.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
    deps = [
        "@protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_proto",
    ],
)

serving_proto_library(
    name = "caffe_source_adapter_proto",
    srcs = ["caffe_source_adapter.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
    deps = [
        ":caffe_session_bundle_config_proto",
    ],
)

genrule(
    name = "pycaffe",
    srcs = caffe_pkg(":pycaffe"),
    outs = ["pycaffe"],
    cmd = '''
        outdir=$(@D)/pycaffe/caffe
        mkdir -p $$outdir
        cp $(SRCS) $$outdir'''
)

cc_library(
    name = "caffe_py_util",
    srcs = ["caffe_py_util.cc"],
    hdrs = ["caffe_py_util.h"],
    deps = caffe_pkg(":lib") + [
        "@caffe_tools//:python_prelude",
        "@org_tensorflow//tensorflow/core:lib",
    ],
    data = if_pycaffe([":pycaffe"])
)

cc_library(
    name = "caffe_signature",
    srcs = ["caffe_signature.cc"],
    hdrs = ["caffe_signature.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":caffe_serving_session",
        "@org_tensorflow//tensorflow/contrib/session_bundle:manifest_proto_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "caffe_session_bundle",
    srcs = ["caffe_session_bundle.cc"],
    hdrs = ["caffe_session_bundle.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@protobuf//:cc_wkt_protos",
        "@protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        ":caffe_serving_session",
    ],
)

cc_library(
    name = "simple_thread_sink",
    srcs = ["simple_thread_sink.cc"],
    hdrs = ["simple_thread_sink.h"],
    linkstatic = 1,
)

cc_library(
    name = "caffe_serving_session",
    srcs = ["caffe_serving_session.cc"],
    hdrs = ["caffe_serving_session.h"],
    deps = caffe_pkg(":lib") + [
        "@caffe_tools//:openblas_prelude",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        ":simple_thread_sink",
        ":caffe_py_util",
        "//tensorflow_serving/servables/tensorflow:serving_session",
    ],
)

cc_library(
    name = "caffe_session_bundle_factory",
    srcs = ["caffe_session_bundle_factory.cc"],
    hdrs = ["caffe_session_bundle_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
        ":caffe_session_bundle_config_proto",
        ":caffe_session_bundle",
        ":caffe_serving_session",
        ":caffe_py_util",
        "//tensorflow_serving/batching:batching_session",
        "//tensorflow_serving/batching:shared_batch_scheduler",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/resources:resources_proto",
    ],
)

cc_test(
    name = "caffe_session_bundle_factory_test",
    srcs = ["caffe_session_bundle_factory_test.cc"],
    data = [
        "//tensorflow_serving/servables/caffe/test_data:mnist_pretrained_caffe",
        "//tensorflow_serving/servables/caffe/test_data:py_layers"
    ],
    deps = [
        "//tensorflow_serving/servables/caffe/test_data:mnist_sample",
        "@protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        ":caffe_session_bundle_config_proto",
        ":caffe_session_bundle_factory",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/test_util",
    ],
)

cc_library(
    name = "caffe_source_adapter",
    srcs = ["caffe_source_adapter.cc"],
    hdrs = ["caffe_source_adapter.h"],
    visibility = [
        "//visibility:public"
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        ":caffe_source_adapter_proto",
        ":caffe_session_bundle",
        ":caffe_session_bundle_factory",
        "//tensorflow_serving/core:simple_loader",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
    ],
)

cc_test(
    name = "caffe_source_adapter_test",
    srcs = ["caffe_source_adapter_test.cc"],
    data = ["//tensorflow_serving/servables/caffe/test_data:mnist_pretrained_caffe"],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        ":caffe_source_adapter",
        ":caffe_source_adapter_proto",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:servable_data",
        "//tensorflow_serving/core/test_util:source_adapter_test_util",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/resources:resource_values",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util:any_ptr",
    ],
)

cc_library(
    name = "caffe_simple_servers",
    srcs = ["caffe_simple_servers.cc"],
    hdrs = ["caffe_simple_servers.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        ":caffe_source_adapter",
        ":caffe_source_adapter_proto",
        ":caffe_session_bundle",
        "//tensorflow_serving/core:aspired_versions_manager_builder",
        "//tensorflow_serving/core:eager_unload_policy",
        "//tensorflow_serving/core:loader",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:source",
        "//tensorflow_serving/core:source_adapter",
        "//tensorflow_serving/core:storage_path",
        "//tensorflow_serving/core:target",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source_proto",
    ],
)

cc_test(
    name = "caffe_simple_servers_test",
    srcs = ["caffe_simple_servers_test.cc"],
    data = ["//tensorflow_serving/servables/caffe/test_data:mnist_pretrained_caffe"],
    linkstatic = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        ":caffe_simple_servers",
        ":caffe_session_bundle",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "//tensorflow_serving/util:unique_ptr_with_deps",
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
        ":caffe_signature",
        ":caffe_session_bundle"
    ],
)