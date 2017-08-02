# Description: Example TensorFlow model export.

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
        ],
    ),
)

py_binary(
    name = "export_half_plus_two",
    srcs = [
        "export_half_plus_two.py",
    ],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "export_bad_half_plus_two",
    srcs = [
        "export_bad_half_plus_two.py",
    ],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

# Note: re-generate these files with :export_half_plus_two whenever model
# changes.
exports_files([
    "half_plus_two/00000123/export.data-00000-of-00001",
    "half_plus_two/00000123/export.index",
    "half_plus_two/00000123/export.meta",
    "saved_model_half_plus_three/00000123/saved_model.pb",
    "saved_model_half_plus_three/00000123/assets/foo.txt",
    "saved_model_half_plus_three/00000123/variables/variables.data-00000-of-00001",
    "saved_model_half_plus_three/00000123/variables/variables.index",
    "bad_half_plus_two/00000123/export",
    "bad_half_plus_two/00000123/export.meta",
    "good_model_config.txt",
    "bad_model_config.txt",
    "batching_config.txt",
])
