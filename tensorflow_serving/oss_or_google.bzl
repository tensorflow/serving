"""
Module for build utilities to distiguish different build environment.
"""

# Whether the compilation environment is open source environment.
is_oss = True

# Helper build function.
# Returns the input if is_oss is true.
# Returns empty list otherwise.
def if_oss(a):
    if is_oss:
        return a
    else:
        return []

# Helper build function.
# Returns the input if is_oss is false.
# Returns empty list otherwise.
def if_google(a):
    if is_oss:
        return []
    else:
        return a

# cc_test that is only run in open source environment.
def oss_only_cc_test(name, srcs = [], deps = [], data = [], size = "medium", linkstatic = 0):
    if is_oss:
        return native.cc_test(
            name = name,
            deps = deps,
            srcs = srcs,
            data = data,
            size = size,
            linkstatic = linkstatic,
        )
    else:
        return None
