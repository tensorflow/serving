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

licenses(["notice"])

exports_files(
    [
        "half_plus_two_model_metadata.json",
        "saved_model_half_plus_two_2_versions_metadata.json",
        "saved_model_half_plus_two_tf2_2_versions_metadata.json",
    ],
    visibility = ["//tensorflow_serving:internal"],
)

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
    name = "export_bad_half_plus_two",
    srcs = [
        "export_bad_half_plus_two.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_binary(
    name = "export_counter",
    srcs = [
        "export_counter.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_binary(
    name = "saved_model_half_plus_two",
    srcs = [
        "saved_model_half_plus_two.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["@org_tensorflow//tensorflow/lite/tools/signature:signature_def_utils"],
)

py_binary(
    name = "parse_example_tflite_with_string",
    srcs = [
        "parse_example_tflite.py",
    ],
    main = "parse_example_tflite.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["@org_tensorflow//tensorflow/lite/tools/signature:signature_def_utils"],
)

filegroup(
    name = "saved_model_half_plus_two_cpu",
    srcs = glob(
        ["saved_model_half_plus_two_cpu/**"],
    ),
)

filegroup(
    name = "saved_model_half_plus_two_tf2_cpu",
    srcs = glob(
        ["saved_model_half_plus_two_tf2_cpu/**"],
    ),
)

filegroup(
    name = "saved_model_half_plus_two_mlmd",
    srcs = glob(
        ["saved_model_half_plus_two_mlmd/**"],
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
    name = "saved_model_half_plus_two_tflite_with_sigdef",
    srcs = glob(
        ["saved_model_half_plus_two_tflite_with_sigdef/**"],
    ),
)

filegroup(
    name = "saved_model_half_plus_two_gpu_trt",
    srcs = glob(
        ["saved_model_half_plus_two_gpu_trt/**"],
    ),
)

filegroup(
    name = "tf_text_regression",
    srcs = glob(
        ["tf_text_regression/**"],
    ),
)

filegroup(
    name = "mobilenet_v1_quant_tflite",
    srcs = glob(
        ["mobilenet_v1_quant_tflite/**"],
    ),
)

filegroup(
    name = "parse_example_tflite",
    srcs = glob(
        ["parse_example_tflite/**"],
    ),
)

exports_files([
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
