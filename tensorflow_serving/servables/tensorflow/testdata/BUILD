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

filegroup(
    name = "saved_model_half_plus_two_2_versions",
    srcs = glob(
        ["saved_model_half_plus_two_2_versions/**/*"],
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
    python_version = "PY2",
    srcs_version = "PY2AND3",
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
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "export_counter",
    srcs = [
        "export_counter.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "saved_model_half_plus_two",
    srcs = [
        "saved_model_half_plus_two.py",
    ],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "saved_model_half_plus_two_cpu",
    srcs = glob(
        ["saved_model_half_plus_two_cpu/**"],
    ),
)

filegroup(
    name = "saved_model_half_plus_two_gpu",
    srcs = glob(
        ["saved_model_half_plus_two_gpu/**"],
    ),
)

filegroup(
    name = "saved_model_half_plus_two_mkl",
    srcs = glob(
        ["saved_model_half_plus_two_mkl/**"],
    ),
)

filegroup(
    name = "saved_model_half_plus_two_tflite",
    srcs = glob(
        ["saved_model_half_plus_two_tflite/**"],
    ),
)

filegroup(
    name = "saved_model_half_plus_two_gpu_trt",
    srcs = glob(
        ["saved_model_half_plus_two_gpu_trt/**"],
    ),
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
    "saved_model_counter/00000123/saved_model.pb",
    "saved_model_counter/00000123/variables/variables.data-00000-of-00001",
    "saved_model_counter/00000123/variables/variables.index",
    "good_model_config.txt",
    "bad_model_config.txt",
    "batching_config.txt",
    "monitoring_config.txt",
])
