"""Build extensions for TensorFlow Serving."""

load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:py_library.bzl", "py_library")

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
        has_services = False,  # buildifier: disable=unused-variable
        deps = [],
        visibility = None,
        testonly = 0,
        cc_grpc_version = None,  # buildifier: disable=unused-variable
        **kwargs):
    """Builds a proto library and corresponding cc_proto_library.

    Args:
      name: The name of the proto_library.
      srcs: The .proto files.
      has_services: True if the proto file contains services.
      deps: Dependencies.
      visibility: Visibility of the targets.
      testonly: Whether the targets are testonly.
      cc_grpc_version: The gRPC version to use for cc_grpc_library.
      **kwargs: Passed to proto_library.
    """

    # Map common deps to their proto_library targets
    pl_deps = []
    for dep in deps:
        if dep == "//google/protobuf:any":
            pass  # Provided by well known protos below
        elif dep.endswith(":cc_wkt_protos") or dep.endswith(":any_cc_proto"):
            pass  # Provided by well known protos below
        elif dep == "//google/protobuf:wrappers":
            pass  # Provided by well known protos below
        elif dep == "//google/protobuf:timestamp":
            pass  # Provided by well known protos below
        elif dep == "//google/protobuf:duration":
            pass  # Provided by well known protos below
        elif dep == "//google/protobuf:empty":
            pass  # Provided by well known protos below
        elif dep.endswith("_cc"):
            pl_deps.append(dep[:-3])
        elif dep.startswith(":") and dep.endswith("_proto"):
            pl_deps.append(dep + "_pl")
        elif (dep.startswith("@org_tensorflow//tensorflow_serving") or dep.startswith("//tensorflow_serving")) and dep.endswith("_proto"):
            pl_deps.append(dep + "_pl")
        else:
            pl_deps.append(dep)

    pl_deps.extend([
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:api_proto",
        "@com_google_protobuf//:descriptor_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:empty_proto",
        "@com_google_protobuf//:field_mask_proto",
        "@com_google_protobuf//:source_context_proto",
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:type_proto",
        "@com_google_protobuf//:wrappers_proto",
    ])

    # Remove duplicates just in case
    pl_deps_unique = []
    for dep in pl_deps:
        if dep not in pl_deps_unique:
            pl_deps_unique.append(dep)
    pl_deps = pl_deps_unique

    proto_library(
        name = name + "_pl",
        srcs = srcs,
        deps = pl_deps,
        visibility = visibility,
        testonly = testonly,
        **kwargs
    )

    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs,
        testonly = testonly,
    )

    cc_proto_library(
        name = name,
        deps = [name + "_pl"],
        testonly = testonly,
        visibility = visibility,
    )

    # For compatibility with Google-internal naming conventions
    if has_services:
        cc_grpc_library(
            name = name[:-len("_proto")] + "_cc_grpc_proto",
            srcs = [name + "_pl"],
            deps = [name],
            grpc_only = True,
            visibility = visibility,
            testonly = testonly,
        )
        cc_library(
            name = name[:-len("_proto")] + "_cc_proto",
            deps = [name, name[:-len("_proto")] + "_cc_grpc_proto"],
            visibility = visibility,
            testonly = testonly,
        )
    else:
        native.alias(
            name = name[:-len("_proto")] + "_cc_proto",
            actual = name,
            testonly = testonly,
            visibility = visibility,
        )

def serving_go_grpc_library(**kwargs):  # pylint: disable=unused-argument
    """Build the Go gRPC bindings for a service. Not yet implemented."""
    return

def serving_proto_library_py(name, proto_library, srcs = [], deps = [], visibility = None, testonly = 0):  # pylint: disable=unused-argument
    py_proto_name = name + "_gen"
    py_proto_library(
        name = py_proto_name,
        deps = [":" + proto_library + "_pl"] if not proto_library.startswith("//") else [proto_library + "_pl"],
        visibility = visibility,
        testonly = testonly,
    )

    # Filter out deps that might be cc_proto_library or proto_library
    # Actually, in tensorflow_serving, deps passed to serving_proto_library_py are Python targets.
    py_deps = [":" + py_proto_name]
    for dep in deps:
        if dep == "//google/protobuf:any":
            py_deps.append("@com_google_protobuf//:protobuf_python")
        elif dep.endswith(":cc_wkt_protos") or dep.endswith(":any_cc_proto"):
            py_deps.append("@com_google_protobuf//:protobuf_python")
        else:
            py_deps.append(dep)

    py_library(
        name = name,
        srcs = [":" + py_proto_name],
        deps = py_deps,
        visibility = visibility,
        testonly = testonly,
    )

def serving_tensorflow_proto_dep(dep):
    """Rename for deps onto tensorflow protos in serving_proto_library targets.
    """
    return dep

def oss_only_cc_test(name, srcs = [], deps = [], data = [], size = "medium", linkstatic = 0):
    """cc_test that is only run in open source environment."""
    return cc_test(
        name = name,
        deps = deps,
        srcs = srcs,
        data = data,
        size = size,
        linkstatic = linkstatic,
    )
