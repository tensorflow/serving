# Description: Tensorflow Serving.
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(default_visibility = ["//tensorflow_serving:internal"])

licenses(["notice"])

exports_files(["LICENSE"])

# open source marker; do not remove

package_group(
    name = "internal",
    packages = ["//tensorflow_serving/..."],
)

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

bool_flag(
    name = "enable_plugins_support",
    build_setting_default = False,
)

config_setting(
    name = "with_plugins_support_define",
    define_values = {"with_plugins_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_plugins_support_flag",
    flag_values = {":enable_plugins_support": "True"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "with_plugins_support",
    match_any = [
        ":with_plugins_support_define",
        ":with_plugins_support_flag",
    ],
    visibility = ["//visibility:public"],
)
