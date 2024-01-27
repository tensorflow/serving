# Description: Tensorflow Serving.

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


config_setting(
    name = "with_plugins_support",
    define_values = {"with_plugins_support": "true"},
    visibility = ["//visibility:public"],
)
