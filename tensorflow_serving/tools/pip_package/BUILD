# Description: Tensorflow Serving pip package.

licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "setup.py",

        # Python scripts needed for the Python TF Serving API
        "//tensorflow_serving/apis:classification_proto_py_pb2",
        "//tensorflow_serving/apis:get_model_metadata_proto_py_pb2",
        "//tensorflow_serving/apis:get_model_status_proto_py_pb2",
        "//tensorflow_serving/apis:inference_proto_py_pb2",
        "//tensorflow_serving/apis:input_proto_py_pb2",
        "//tensorflow_serving/apis:model_management_proto_py_pb2",
        "//tensorflow_serving/apis:model_proto_py_pb2",
        "//tensorflow_serving/apis:model_service_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_log_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:regression_proto_py_pb2",
        "//tensorflow_serving/apis:session_service_proto_py_pb2",
        "//tensorflow_serving/config:log_collector_config_proto_py_pb2",
        "//tensorflow_serving/config:logging_config_proto_py_pb2",
        "//tensorflow_serving/config:model_server_config_proto_py_pb2",
        "//tensorflow_serving/sources/storage_path:file_system_storage_path_source_proto_py_pb2",
        "//tensorflow_serving/util:status_proto_py_pb2",
        "//tensorflow_serving/core:logging_proto_py_pb2",
    ],
)
