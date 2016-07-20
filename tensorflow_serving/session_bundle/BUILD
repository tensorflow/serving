# Description: Tensorflow Serving session bundle.
#
# All C++ targets simply forward to the equivalent
# tensorflow/contrib/session_bundle targets.
# Python code will soon be updated in a similar way.
#
# IMPORTANT: All code in tensorflow_serving/session_bundle will soon be deleted
# In favor of the copy in tensorflow/contrib/session_bundle. All new code
# should use the new directory instead.

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
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow/contrib/session_bundle:manifest_proto_py_pb2",
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
        "//third_party/py/tensorflow",
        "@org_tensorflow//tensorflow/contrib/session_bundle:manifest_proto_py_pb2",
    ],
)

py_library(
    name = "gc",
    srcs = ["gc.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/tensorflow",
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
        "//third_party/py/tensorflow",
    ],
)

# Proto header forwarding to tensorflow/contrib/session_bundle:manifest_proto.
cc_library(
    name = "manifest_proto",
    hdrs = [
        "manifest.pb.h",
        "manifest.proto.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["@org_tensorflow//tensorflow/contrib/session_bundle:manifest_proto"],
)

# Header forwarding to tensorflow/contrib/session_bundle:session_bundle.
cc_library(
    name = "session_bundle",
    hdrs = ["session_bundle.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":signature",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
    ],
)

# Header forwarding to tensorflow/contrib/session_bundle:session_bundle_lite.
#
# This is a lite version of the session_bundle target that does not link in any
# Tensorflow ops in order to minimize its size. Clients using this should link
# any required ops manually.
cc_library(
    name = "session_bundle_lite",
    hdrs = ["session_bundle.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/contrib/session_bundle:session_bundle_lite",
    ],
)

# Header forwarding to tensorflow/contrib/session_bundle:signature.
cc_library(
    name = "signature",
    hdrs = ["signature.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/contrib/session_bundle:signature",
    ],
)

# Header forwarding to tensorflow/contrib/session_bundle:signature_lite.
#
# This is a lite version of the signature target that does not link in any
# Tensorflow ops in order to minimize its size. Clients using this should
# link any required ops manually.
cc_library(
    name = "signature_lite",
    hdrs = ["signature.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/contrib/session_bundle:signature_lite",
    ],
)
