# Description: Tensorflow Serving.

load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = ["//tensorflow_serving:license"],
    default_visibility = ["//tensorflow_serving:internal"],
)

license(
    name = "license",
    package_name = "tensorflow_serving",
)

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
