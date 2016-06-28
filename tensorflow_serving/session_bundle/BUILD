# Description: Tensorflow Serving session bundle.

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
            "g3doc/sitemap.md",
        ],
    ),
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_library(
    name = "exporter",
    srcs = ["exporter.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":constants",
        ":gc",
        ":manifest_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "session_bundle_py",
    srcs = ["session_bundle.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":constants",
        ":exporter",
        ":manifest_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

py_test(
    name = "exporter_test",
    size = "small",
    srcs = [
        "exporter_test.py",
    ],
    deps = [
        ":constants",
        ":exporter",
        ":gc",
        ":manifest_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "session_bundle_py_test",
    size = "small",
    srcs = [
        "session_bundle_test.py",
    ],
    data = [
        "//tensorflow_serving/session_bundle/example:half_plus_two",
    ],
    main = "session_bundle_test.py",
    deps = [
        ":manifest_proto_py",
        ":session_bundle_py",
        "//tensorflow_serving/test_util:test_util_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
        "@protobuf//:protobuf_python",
    ],
)

py_library(
    name = "gc",
    srcs = ["gc.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "gc_test",
    srcs = [
        "gc_test.py",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":gc",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

load("//tensorflow_serving:serving.bzl", "serving_proto_library")
load("//tensorflow_serving:serving.bzl", "serving_proto_library_py")

serving_proto_library(
    name = "manifest_proto",
    srcs = ["manifest.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

serving_proto_library_py(
    name = "manifest_proto_py",
    srcs = ["manifest.proto"],
    visibility = ["//visibility:public"],
    deps = ["@org_tensorflow//tensorflow/core:protos_all_py"],
)

cc_library(
    name = "session_bundle",
    srcs = ["session_bundle.cc"],
    hdrs = ["session_bundle.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":manifest_proto",
        ":signature",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow_opensource",
    ],
)

# This is a lite version of the session_bundle target that does not link in any
# Tensorflow ops in order to minimize its size. Clients using this should link
# any required ops manually.
cc_library(
    name = "session_bundle_lite",
    srcs = ["session_bundle.cc"],
    hdrs = ["session_bundle.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":manifest_proto",
        ":signature_lite",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

filegroup(
    name = "session_bundle_srcs",
    srcs = [
        "session_bundle.cc",
        "session_bundle.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "signature_srcs",
    srcs = [
        "signature.cc",
        "signature.h",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "session_bundle_test",
    size = "small",
    srcs = ["session_bundle_test.cc"],
    data = [
        "//tensorflow_serving/session_bundle/example:half_plus_two",
    ],
    # Link in all registered kernels.
    linkstatic = 1,
    deps = [
        ":session_bundle",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@protobuf//:cc_wkt_protos",
    ],
)

cc_library(
    name = "signature",
    srcs = ["signature.cc"],
    hdrs = ["signature.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":manifest_proto",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:tensorflow_opensource",
    ],
)

# This is a lite version of the signature target that does not link in any
# Tensorflow ops in order to minimize its size. Clients using this should
# link any required ops manually.
cc_library(
    name = "signature_lite",
    srcs = ["signature.cc"],
    hdrs = ["signature.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":manifest_proto",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_test(
    name = "signature_test",
    size = "small",
    srcs = ["signature_test.cc"],
    deps = [
        ":manifest_proto",
        ":signature",
        "//tensorflow_serving/core/test_util:test_main",
        "//tensorflow_serving/test_util",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow_opensource",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@protobuf//:cc_wkt_protos",
        "@protobuf//:protobuf_lite",
    ],
)
