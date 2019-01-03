# Description: Tests for Docker images.
# These tests bring up a docker container based on a locally installed
# image (passed via an argument), and verify that the functionality of
# the docker image is intact. The tests require an environment to have
# docker installed.
#
# Tests are local and manual as they require docker and the image to be
# installed.

licenses(["notice"])  # Apache 2.0

sh_library(
    name = "docker_test_lib",
    srcs = ["docker_test_lib.sh"],
)

sh_test(
    name = "unittest_dockerfile",
    size = "small",
    srcs = ["dockerfile_test.sh"],
    args = ["tensorflow/serving:latest"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_cpu",
    ],
    tags = [
        "exclusive",
        "local",
        "manual",
    ],
    deps = [":docker_test_lib"],
)

sh_test(
    name = "unittest_dockerfile_devel",
    size = "small",
    srcs = ["dockerfile_devel_test.sh"],
    args = ["tensorflow/serving:latest-devel"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_cpu",
    ],
    tags = [
        "exclusive",
        "local",
        "manual",
    ],
    deps = [":docker_test_lib"],
)

sh_test(
    name = "unittest_dockerfile_mkl",
    size = "small",
    srcs = ["dockerfile_mkl_test.sh"],
    args = ["tensorflow/serving:latest-mkl"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_mkl",
    ],
    tags = [
        "exclusive",
        "local",
        "manual",
    ],
    deps = [":docker_test_lib"],
)

sh_test(
    name = "unittest_dockerfile_devel_mkl",
    size = "small",
    srcs = ["dockerfile_devel_mkl_test.sh"],
    args = ["tensorflow/serving:latest-devel-mkl"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_mkl",
    ],
    tags = [
        "exclusive",
        "local",
        "manual",
    ],
    deps = [":docker_test_lib"],
)

sh_test(
    name = "unittest_dockerfile_gpu",
    size = "medium",
    srcs = ["dockerfile_gpu_test.sh"],
    args = ["tensorflow/serving:latest-gpu"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_gpu",
    ],
    tags = [
        "exclusive",
        "local",
        "manual",
    ],
    deps = [":docker_test_lib"],
)

sh_test(
    name = "unittest_dockerfile_devel_gpu",
    size = "medium",
    srcs = ["dockerfile_devel_gpu_test.sh"],
    args = ["tensorflow/serving:latest-devel-gpu"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_gpu",
    ],
    tags = [
        "exclusive",
        "local",
        "manual",
    ],
    deps = [":docker_test_lib"],
)

sh_test(
    name = "unittest_dockerfile_gpu_trt",
    size = "medium",
    srcs = ["dockerfile_gpu_trt_test.sh"],
    args = ["tensorflow/serving:latest-gpu"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_gpu_trt",
    ],
    tags = [
        "exclusive",
        "local",
        "manual",
    ],
    deps = [":docker_test_lib"],
)

sh_test(
    name = "unittest_dockerfile_devel_gpu_trt",
    size = "medium",
    srcs = ["dockerfile_devel_gpu_trt_test.sh"],
    args = ["tensorflow/serving:latest-devel-gpu"],
    data = [
        "//tensorflow_serving/servables/tensorflow/testdata:saved_model_half_plus_two_gpu_trt",
    ],
    tags = [
        "exclusive",
        "local",
        "manual",
    ],
    deps = [":docker_test_lib"],
)
