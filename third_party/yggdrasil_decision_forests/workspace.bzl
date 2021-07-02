"""Yggdrasil Decision Forests project."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def deps():
    http_archive(
        name = "ydf",
        urls = ["https://github.com/google/yggdrasil-decision-forests/archive/refs/heads/main.zip"],
        strip_prefix = "yggdrasil-decision-forests-main",
    )