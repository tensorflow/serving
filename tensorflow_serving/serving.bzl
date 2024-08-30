load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@local_tsl//tsl/platform/default:build_config.bzl", "py_proto_library")

def if_oss(oss_value):
    """Returns oss_value if in OSS build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return oss_value

def if_google(
        google_value):  # @unused
    """Returns google_value if in Google build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    return []

def serving_proto_library(
        name,
        srcs = [],
        has_services = False,  # pylint: disable=unused-argument
        deps = [],
        visibility = None,
        testonly = 0,
        cc_grpc_version = None):
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs,
        testonly = testonly,
    )

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True

    # For compatibility with Google-internal naming conventions
    native.alias(
        name = name[:-len("_proto")] + "_cc_proto",
        actual = name,
        testonly = testonly,
        visibility = visibility,
    )

    cc_proto_library(
        name = name,
        srcs = srcs,
        deps = deps,
        cc_libs = ["@com_google_protobuf//:protobuf"],
        protoc = "@com_google_protobuf//:protoc",
        default_runtime = "@com_google_protobuf//:protobuf",
        use_grpc_plugin = use_grpc_plugin,
        testonly = testonly,
        visibility = visibility,
    )

def serving_go_grpc_library(**kwargs):  # pylint: disable=unused-argument
    """Build the Go gRPC bindings for a service. Not yet implemented."""
    return

def serving_proto_library_py(name, proto_library, srcs = [], deps = [], visibility = None, testonly = 0):  # pylint: disable=unused-argument
    py_proto_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        deps = ["@com_google_protobuf//:protobuf_python"] + deps,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        visibility = visibility,
        testonly = testonly,
    )

def serving_tensorflow_proto_dep(dep):
    """Rename for deps onto tensorflow protos in serving_proto_library targets.
    """
    return "{}_cc".format(dep)

def oss_only_cc_test(name, srcs = [], deps = [], data = [], size = "medium", linkstatic = 0):
    """cc_test that is only run in open source environment."""
    return native.cc_test(
        name = name,
        deps = deps,
        srcs = srcs,
        data = data,
        size = size,
        linkstatic = linkstatic,
    )
