# libevent (libevent.org) library.
# from https://github.com/libevent/libevent

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE"])

include_files = [
    "libevent/include/evdns.h",
    "libevent/include/event.h",
    "libevent/include/evhttp.h",
    "libevent/include/evrpc.h",
    "libevent/include/evutil.h",
    "libevent/include/event2/buffer.h",
    "libevent/include/event2/bufferevent_struct.h",
    "libevent/include/event2/event.h",
    "libevent/include/event2/http_struct.h",
    "libevent/include/event2/rpc_struct.h",
    "libevent/include/event2/buffer_compat.h",
    "libevent/include/event2/dns.h",
    "libevent/include/event2/event_compat.h",
    "libevent/include/event2/keyvalq_struct.h",
    "libevent/include/event2/tag.h",
    "libevent/include/event2/bufferevent.h",
    "libevent/include/event2/dns_compat.h",
    "libevent/include/event2/event_struct.h",
    "libevent/include/event2/listener.h",
    "libevent/include/event2/tag_compat.h",
    "libevent/include/event2/bufferevent_compat.h",
    "libevent/include/event2/dns_struct.h",
    "libevent/include/event2/http.h",
    "libevent/include/event2/rpc.h",
    "libevent/include/event2/thread.h",
    "libevent/include/event2/event-config.h",
    "libevent/include/event2/http_compat.h",
    "libevent/include/event2/rpc_compat.h",
    "libevent/include/event2/util.h",
    "libevent/include/event2/visibility.h",
]

lib_files = [
    "libevent/lib/libevent.a",
    "libevent/lib/libevent_core.a",
    "libevent/lib/libevent_extra.a",
    "libevent/lib/libevent_pthreads.a",
]

genrule(
    name = "libevent-srcs",
    srcs = glob(["**"]) + ["@sysroot_linux_x86_64//:sysroot"],
    outs = include_files + lib_files,
    cmd = "\n".join([
        "export INSTALL_DIR=$$(pwd)/$(@D)/libevent",
        "export ORIG_DIR=$$(pwd) # force rebuild 1",
        "export CC_PATH=$(CC)",
        "if [[ $$CC_PATH != /* ]]; then CC_PATH=$$ORIG_DIR/$$CC_PATH; fi",
        "export REAL_CLANG=external/llvm18_linux_x86_64/bin/clang",
        "if [[ $$REAL_CLANG != /* ]]; then REAL_CLANG=$$ORIG_DIR/$$REAL_CLANG; fi",
        "export GCC_PATH=$$REAL_CLANG",
        "export REAL_AR=external/llvm18_linux_x86_64/bin/llvm-ar",
        "if [[ $$REAL_AR != /* ]]; then REAL_AR=$$ORIG_DIR/$$REAL_AR; fi",
        "export SYSROOT_DIR=$$(ls -d $$ORIG_DIR/external/sysroot_linux_x86_64_glibc_* | head -n 1)",
        "export TMP_DIR=$$(mktemp -d -t libevent.XXXXXX)",
        "mkdir -p $$TMP_DIR",
        "cp -R $$ORIG_DIR/external/com_github_libevent_libevent/* $$TMP_DIR",
        "cd $$TMP_DIR",
        "./autogen.sh",
        "CC=$$CC_PATH AR=$$REAL_AR ./configure --prefix=$$INSTALL_DIR CFLAGS=\"-fPIC --sysroot=$$SYSROOT_DIR\" CXXFLAGS=\"-fPIC --sysroot=$$SYSROOT_DIR\" LDFLAGS=\"--sysroot=$$SYSROOT_DIR\" --enable-shared=no --disable-openssl || (cat config.log && false)",
        "make install CC=$$CC_PATH AR=$$REAL_AR",
        "rm -rf $$TMP_DIR",
    ]),
    local = 1,
    toolchains = ["@bazel_tools//tools/cpp:current_cc_toolchain"],
)

cc_library(
    name = "libevent",
    srcs = [
        "libevent/lib/libevent.a",
        "libevent/lib/libevent_pthreads.a",
    ],
    hdrs = include_files,
    includes = ["libevent/include"],
    linkopts = ["-lpthread"],
    linkstatic = 1,
)

filegroup(
    name = "libevent-files",
    srcs = include_files + lib_files,
)
