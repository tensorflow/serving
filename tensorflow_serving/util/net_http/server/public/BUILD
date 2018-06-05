# Description: a lightweight http server and related utils to support Web clients

package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "http_server_api",
    hdrs = [
        "httpserver_interface.h",
        "response_code_enum.h",
        "server_request_interface.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "http_server",
    hdrs = [
        "httpserver.h",
    ],
    deps = [
        ":http_server_api",
        "//tensorflow_serving/util/net_http/server/internal:evhttp_server",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
    ],
)
