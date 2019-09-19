"""
Module for build utilities to distiguish different tensorflow versions.
"""

load("@org_tensorflow//tensorflow:tensorflow.bzl", "VERSION_MAJOR")

def if_v2(a):
    if VERSION_MAJOR == "2":
        return a
    else:
        return []

def if_not_v2(a):
    if VERSION_MAJOR == "2":
        return []
    else:
        return a
