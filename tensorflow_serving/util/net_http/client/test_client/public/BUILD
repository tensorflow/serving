# Description: APIs for experimental testing of net_http server instances

package(
    default_visibility = [
        ":http_client_users",
        "//tensorflow_serving/util/net_http:__subpackages__",
    ],
)

package_group(
    name = "http_client_users",
    packages = [
        "//third_party/ecclesia/...",
    ],
)

licenses(["notice"])

cc_library(
    name = "http_client_api",
    srcs = [],
    hdrs = [
        "httpclient_interface.h",
    ],
    deps = [
        "//tensorflow_serving/util/net_http/public:shared_files",
        "//tensorflow_serving/util/net_http/server/public:http_server_api",
    ],
)

cc_library(
    name = "http_client",
    hdrs = [
        "httpclient.h",
    ],
    deps = [
        ":http_client_api",
        "//tensorflow_serving/util/net_http/client/test_client/internal:evhttp_client",
        "@com_google_absl//absl/memory",
    ],
)
