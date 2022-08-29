load("//tensorflow_serving:oss_or_google.bzl", "if_google", "if_oss")

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
)

filegroup(
    name = "session_bundle_half_plus_two",
    srcs = glob([
        "testdata/half_plus_two/**",
    ]),
)

cc_library(
    name = "manifest_proto_header",
    hdrs = ["manifest_proto.h"],
    deps = ["//tensorflow_serving/util:oss_or_google"] + if_oss([
        "//tensorflow_serving/session_bundle/oss:manifest_proto_cc",
    ]) + if_google(["//tensorflow_serving/session_bundle/google:manifest_proto_cc"]),
)

cc_library(
    name = "session_bundle",
    hdrs = ["session_bundle.h"],
    deps = ["//tensorflow_serving/util:oss_or_google"] + if_oss([
        "//tensorflow_serving/session_bundle/oss:session_bundle",
    ]) + if_google(["@org_tensorflow//tensorflow/contrib/session_bundle"]),
)

cc_library(
    name = "session_bundle_util_header",
    hdrs = ["session_bundle_util.h"],
    deps = [
        ":manifest_proto_header",
        ":session_bundle",
        "@org_tensorflow//tensorflow/cc/saved_model:loader",
    ],
)

cc_library(
    name = "session_bundle_util",
    deps = if_oss(["//tensorflow_serving/session_bundle/oss:session_bundle_util"]) + if_google(["//tensorflow_serving/session_bundle/google:session_bundle_util"]),
)
