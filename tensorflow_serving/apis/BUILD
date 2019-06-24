# Description: Tensorflow Serving APIs.

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
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

load("//tensorflow_serving:serving.bzl", "serving_proto_library")
load("//tensorflow_serving:serving.bzl", "serving_proto_library_py")
load("//tensorflow_serving:serving.bzl", "serving_go_grpc_library")
load("@org_tensorflow//tensorflow/core:platform/default/build_config.bzl", "tf_jspb_proto_library", "tf_pyclif_proto_library")

serving_proto_library(
    name = "get_model_metadata_proto",
    srcs = ["get_model_metadata.proto"],
    cc_api_version = 2,
    deps = [
        ":model_proto",
        "@com_google_protobuf//:cc_wkt_protos",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

serving_proto_library_py(
    name = "get_model_metadata_proto_py_pb2",
    srcs = ["get_model_metadata.proto"],
    proto_library = "get_model_metadata_proto",
    deps = [
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

serving_proto_library(
    name = "input_proto",
    srcs = ["input.proto"],
    cc_api_version = 2,
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

serving_proto_library_py(
    name = "input_proto_py_pb2",
    srcs = ["input.proto"],
    proto_library = "input_proto",
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

tf_pyclif_proto_library(
    name = "input_pyclif",
    proto_lib = ":input_proto",
    proto_srcfile = "input.proto",
)

tf_jspb_proto_library(
    name = "input_jspb_proto",
    deps = [":input_proto"],
)

serving_proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
    cc_api_version = 2,
    deps = [
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

serving_proto_library_py(
    name = "model_proto_py_pb2",
    srcs = ["model.proto"],
    proto_library = "model_proto",
    deps = [],
)

tf_pyclif_proto_library(
    name = "model_pyclif",
    proto_lib = ":model_proto",
    proto_srcfile = "model.proto",
)

tf_jspb_proto_library(
    name = "model_jspb_proto",
    deps = [":model_proto"],
)

serving_proto_library(
    name = "predict_proto",
    srcs = ["predict.proto"],
    cc_api_version = 2,
    deps = [
        ":model_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

serving_proto_library_py(
    name = "predict_proto_py_pb2",
    srcs = ["predict.proto"],
    proto_library = "predict_proto",
    deps = [
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

tf_jspb_proto_library(
    name = "predict_jspb_proto",
    deps = [":predict_proto"],
)

serving_proto_library(
    name = "prediction_log_proto",
    srcs = ["prediction_log.proto"],
    cc_api_version = 2,
    deps = [
        ":classification_proto",
        ":inference_proto",
        ":predict_proto",
        ":regression_proto",
        ":session_service_proto",
        "//tensorflow_serving/core:logging_proto",
    ],
)

serving_proto_library_py(
    name = "prediction_log_proto_py_pb2",
    srcs = ["prediction_log.proto"],
    proto_library = "prediction_log_proto",
    deps = [
        ":classification_proto_py_pb2",
        ":inference_proto_py_pb2",
        ":predict_proto_py_pb2",
        ":regression_proto_py_pb2",
        ":session_service_proto_py_pb2",
        "//tensorflow_serving/core:logging_proto_py_pb2",
    ],
)

serving_proto_library(
    name = "prediction_service_proto",
    srcs = ["prediction_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    deps = [
        ":classification_proto",
        ":get_model_metadata_proto",
        ":inference_proto",
        ":predict_proto",
        ":regression_proto",
    ],
)

py_library(
    name = "prediction_service_proto_py_pb2",
    srcs = [
        "prediction_service_pb2.py",
        "prediction_service_pb2_grpc.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":classification_proto_py_pb2",
        ":get_model_metadata_proto_py_pb2",
        ":inference_proto_py_pb2",
        ":predict_proto_py_pb2",
        ":regression_proto_py_pb2",
    ],
)

tf_pyclif_proto_library(
    name = "predict_pyclif",
    proto_lib = ":predict_proto",
    proto_srcfile = "predict.proto",
)

serving_go_grpc_library(
    name = "prediction_service_grpc",
    srcs = [":prediction_service_proto"],
    deps = [":prediction_service_go_proto"],
)

serving_proto_library(
    name = "model_management_proto",
    srcs = ["model_management.proto"],
    cc_api_version = 2,
    deps = [
        "//tensorflow_serving/config:model_server_config_proto",
        "//tensorflow_serving/util:status_proto",
    ],
)

serving_proto_library_py(
    name = "model_management_proto_py_pb2",
    srcs = ["model_management.proto"],
    proto_library = "model_management_proto",
    deps = [
        "//tensorflow_serving/config:model_server_config_proto_py_pb2",
        "//tensorflow_serving/util:status_proto_py_pb2",
    ],
)

serving_proto_library(
    name = "get_model_status_proto",
    srcs = ["get_model_status.proto"],
    cc_api_version = 2,
    deps = [
        ":model_proto",
        "//tensorflow_serving/util:status_proto",
    ],
)

serving_proto_library_py(
    name = "get_model_status_proto_py_pb2",
    srcs = ["get_model_status.proto"],
    proto_library = "get_model_status_proto",
    deps = [
        ":model_proto_py_pb2",
        "//tensorflow_serving/util:status_proto_py_pb2",
    ],
)

serving_proto_library(
    name = "model_service_proto",
    srcs = ["model_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
    deps = [
        ":get_model_status_proto",
        ":model_management_proto",
    ],
)

py_library(
    name = "model_service_proto_py_pb2",
    srcs = [
        "model_service_pb2.py",
        "model_service_pb2_grpc.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":get_model_status_proto_py_pb2",
        ":model_management_proto_py_pb2",
    ],
)

serving_go_grpc_library(
    name = "model_service_grpc",
    srcs = [":model_service_proto"],
    deps = [":model_service_go_proto"],
)

serving_proto_library(
    name = "classification_proto",
    srcs = ["classification.proto"],
    cc_api_version = 2,
    deps = [
        ":input_proto",
        ":model_proto",
    ],
)

serving_proto_library_py(
    name = "classification_proto_py_pb2",
    srcs = ["classification.proto"],
    proto_library = "classification_proto",
    deps = [
        ":input_proto_py_pb2",
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

tf_pyclif_proto_library(
    name = "classification_pyclif",
    proto_lib = ":classification_proto",
    proto_srcfile = "classification.proto",
)

tf_jspb_proto_library(
    name = "classification_jspb_proto",
    deps = [":classification_proto"],
)

serving_proto_library(
    name = "inference_proto",
    srcs = ["inference.proto"],
    cc_api_version = 2,
    deps = [
        ":classification_proto",
        ":input_proto",
        ":model_proto",
        ":regression_proto",
    ],
)

serving_proto_library_py(
    name = "inference_proto_py_pb2",
    srcs = ["inference.proto"],
    proto_library = "inference_proto",
    deps = [
        ":classification_proto_py_pb2",
        ":input_proto_py_pb2",
        ":model_proto_py_pb2",
        ":regression_proto_py_pb2",
    ],
)

tf_pyclif_proto_library(
    name = "inference_pyclif",
    proto_lib = ":inference_proto",
    proto_srcfile = "inference.proto",
)

serving_proto_library(
    name = "regression_proto",
    srcs = ["regression.proto"],
    cc_api_version = 2,
    deps = [
        ":input_proto",
        ":model_proto",
    ],
)

serving_proto_library_py(
    name = "regression_proto_py_pb2",
    srcs = ["regression.proto"],
    proto_library = "regression_proto",
    deps = [
        ":input_proto_py_pb2",
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

tf_jspb_proto_library(
    name = "regression_jspb_proto",
    deps = [":regression_proto"],
)

serving_proto_library(
    name = "session_service_proto",
    srcs = ["session_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    deps = [
        ":model_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

serving_proto_library_py(
    name = "session_service_proto_py_pb2",
    srcs = ["session_service.proto"],
    proto_library = "session_service_proto",
    deps = [
        ":model_proto_py_pb2",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
)

tf_pyclif_proto_library(
    name = "regression_pyclif",
    proto_lib = ":regression_proto",
    proto_srcfile = "regression.proto",
)

cc_library(
    name = "classifier",
    hdrs = ["classifier.h"],
    deps = [
        ":classification_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "regressor",
    hdrs = ["regressor.h"],
    deps = [
        ":regression_proto",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
