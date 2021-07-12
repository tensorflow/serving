load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow_serving/custom_ops/tfdf:utils.bzl", "rpath_linkopts_to_tensorflow","tf_custom_op_library_external")
load("//tensorflow_serving/custom_ops/tfdf:utils.bzl", "rpath_linkopts_to_tensorflow")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)
### mods
bzl_library(
    name = "utils_bzl",
    srcs = ["utils.bzl"],
)

cc_library(
    name = "canonical_models",
    deps = [
        "@ydf//yggdrasil_decision_forests/model/gradient_boosted_trees",
        "@ydf//yggdrasil_decision_forests/model/random_forest",
        "@ydf//yggdrasil_decision_forests/serving/decision_forest:register_engines",
    ],
    alwayslink = 1,
)

cc_library(
    name = 'tfdf',
    srcs = [
        "op.cc",
	"kernel.cc"
    ],
    alwayslink = 1,
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
	"@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf//yggdrasil_decision_forests/model:abstract_model",
        "@ydf//yggdrasil_decision_forests/model:model_library",
        "@ydf//yggdrasil_decision_forests/utils:compatibility",
        "@ydf//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
        "@ydf//yggdrasil_decision_forests/utils:tensorflow",
	":canonical_models",
    ]
)

tf_custom_op_library_external(
    name = "inference.so",
    deps = [":kernel_and_op"],
)

cc_library(
    name = "op",
    srcs = ["op.cc"],
    linkstatic = 1,
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib"],
    alwayslink = 1,
)

cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_cc_proto",
        "@ydf//yggdrasil_decision_forests/dataset:vertical_dataset",
        "@ydf//yggdrasil_decision_forests/model:abstract_model",
        "@ydf//yggdrasil_decision_forests/model:model_library",
        "@ydf//yggdrasil_decision_forests/utils:compatibility",
        "@ydf//yggdrasil_decision_forests/utils:distribution_cc_proto",
        "@ydf//yggdrasil_decision_forests/utils:status_macros",
        "@ydf//yggdrasil_decision_forests/utils:tensorflow",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "op_py_no_precompile",
    out = "op.py",
    cc_linkopts = rpath_linkopts_to_tensorflow("op"),
    deps = [":op"],
)

cc_library(
    name = "kernel_and_op",
    deps = [
        ":kernel",
        ":op",
        ":canonical_models",
    ],
    alwayslink = 1,
)

py_library(
    name = "api_py",
    srcs = ["api.py"],
    data = [":inference.so"],
    srcs_version = "PY3",
    deps = [
        ":op_py",
        # absl/logging dep,
        # six dep,
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@ydf//yggdrasil_decision_forests/dataset:data_spec_py_proto",
        "@ydf//yggdrasil_decision_forests/model:abstract_model_py_proto",
    ],
)

alias(
    name = "op_py",
    actual = select({
        "//conditions:default": ":op_py_no_precompile",
    }),
)

py_library(
    name = "op_py_precompile",
    srcs = ["op.py"],
    srcs_version = "PY3",
    deps = [
        "@org_tensorflow//tensorflow/python",
    ],
)