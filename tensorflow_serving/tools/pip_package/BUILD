# Description: Tensorflow Serving pip package.

licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "setup.py",

        # Scripts.
        "//tensorflow_serving/apis:get_model_metadata_proto_py_pb2",
        "//tensorflow_serving/apis:input_proto_py_pb2",
        "//tensorflow_serving/apis:model_proto_py_pb2",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "//tensorflow_serving/apis:classification_proto_py_pb2",
        "//tensorflow_serving/apis:inference_proto_py_pb2",
        "//tensorflow_serving/apis:regression_proto_py_pb2",
    ],
)
