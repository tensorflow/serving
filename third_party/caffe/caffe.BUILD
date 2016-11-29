load("@org_tensorflow//tensorflow:tensorflow.bzl", "if_cuda")
load("@caffe_tools//:config.bzl", "if_pycaffe")
load("@local_config_cuda//cuda:platform.bzl",
     "cuda_sdk_version",
     "cudnn_library_path",
    )

package(default_visibility = ["//visibility:public"])

# gets the filepath to protobuf
genquery(
    name = "protobuf-root",
    expression = "@protobuf//:protobuf_lite",
    scope = ["@protobuf//:protobuf_lite"],
    opts = ["--output=location"]
)

genrule(
    name = "pycaffe",
    srcs = [
        "python/caffe/pycaffe.py",
        "@caffe_tools//:pycaffe_overrides"
    ],
    outs = [
        "__init__.py",
        "pycaffe.py",
        "io.py"
    ],
    cmd = "cp $(SRCS) $(@D); touch $(@D)/io.py;"
)

genrule(
    name = "configure",
    message = "Building Caffe (this may take a while)",
    srcs = if_cuda([
        "@local_config_cuda//cuda:include/cudnn.h",
        "@local_config_cuda//cuda:" + cudnn_library_path()
    ]) + [
        ":protobuf-root",
        ":CMakeLists.txt",
        "@protobuf//:protoc",
        "@protobuf//:protobuf_lite"
    ],
    outs = [
        # caffe
        "lib/libcaffe.a",
        "lib/libproto.a",
        "include/caffe/proto/caffe.pb.h",
        # pycaffe
        "lib/_caffe.cpp.o",
        # openblas
        "lib/libopenblas.so.0",
        "include/cblas.h",
        "include/openblas_config.h",
    ],
    cmd =
        '''
        srcdir=$$(pwd);
        workdir=$$(mktemp -d -t tmp.XXXXXXXXXX);
        outdir=$$srcdir/$(@D);

        protobuf_incl=$$(grep -oP "^/\\\S*(?=/)" $(location :protobuf-root))/src;
        protoc=$$srcdir/$(location @protobuf//:protoc);
        protolib=$$srcdir/$$(echo "$(locations @protobuf//:protobuf_lite)" | grep -o "\\\S*/libprotobuf_lite.a"); ''' +

        # extra cmake options during cuda configuration,
        # adopting the tensorflow cuda configuration where
        # sensible.
        if_cuda('''
            cudnn_includes=$(location @local_config_cuda//cuda:include/cudnn.h);
            cudnn_lib=$(location @local_config_cuda//cuda:%s);
            extra_cmake_opts="-DCPU_ONLY:bool=OFF
                              -DUSE_CUDNN:bool=ON
                              -DCUDNN_INCLUDE:path=$$srcdir/$$(dirname $$cudnn_includes)
                              -DCUDNN_LIBRARY:path=$$srcdir/$$cudnn_lib"; ''' % cudnn_library_path(),

            'extra_cmake_opts="-DCPU_ONLY:bool=ON";') +

        # python layers
        if_pycaffe('py_layer=ON;', 'py_layer=OFF;') +

        # configure cmake.
        '''
        pushd $$workdir;
        cmake $$srcdir/$$(dirname $(location :CMakeLists.txt)) \
            -DCMAKE_INSTALL_PREFIX=$$srcdir/$(@D) \
            -DCMAKE_BUILD_TYPE=Release            \
            -DBLAS:string="open"                  \
            -DBUILD_python=$$py_layer             \
            -DBUILD_python_layer=$$py_layer       \
            -DUSE_OPENCV=OFF                      \
            -DBUILD_SHARED_LIBS=OFF               \
            -DUSE_LEVELDB=OFF                     \
            -DUSE_LMDB=OFF                        \
            -DPROTOBUF_INCLUDE_DIR=$$protobuf_incl\
            -DPROTOBUF_PROTOC_EXECUTABLE=$$protoc \
            -DPROTOBUF_LIBRARY=$$protolib         \
            $${extra_cmake_opts}; ''' +

        # build libcaffe.a -- note we avoid building the
        # caffe tools because 1) we don't need them anyway
        # and 2) they will fail to link because only
        # protobuf_lite.a (and not libprotobuf.so) is
        # specified in PROTOBUF_LIBRARY.
        '''
        cmake --build . --target caffe -- -j 4
        cp -r ./lib $$outdir
        cp -r ./include $$outdir; ''' +

        if_pycaffe('''
            # hack; we're not interested in _caffe.so, and it will
            # probably fail to link against libprotobuf_lite.a anyway.
            touch -d '+1 hour' lib/_caffe.so;

            cmake --build . --target pycaffe;
            cp ./python/CMakeFiles/pycaffe.dir/caffe/_caffe.cpp.o $$outdir/lib; ''', '''
            touch $$outdir/lib/_caffe.cpp.o; '''
        ) +

        # openblas (note the full soname is libopenblas.so.0)
        '''
        openblas_incl=$$(grep -oP 'OpenBLAS_INCLUDE_DIR:PATH=\K(.*)' CMakeCache.txt)
        openblas_lib=$$(grep -oP 'OpenBLAS_LIB:FILEPATH=\K(.*)' CMakeCache.txt)

        cp $$openblas_lib $$outdir/lib/libopenblas.so.0
        cp $$openblas_incl/cblas.h $$outdir/include
        
        # copy config header (not always present)
        if [ -f "$$openblas_incl/openblas_config.h" ]; then
            cp $$openblas_incl/openblas_config.h $$outdir/include
        else
            touch $$outdir/include/openblas_config.h
        fi ''' +

        '''
        # clean up
        popd;
        # rm -rf $$workdir; ''',
)

# Note: Bazel will ignore `alwayslink=1` for *.a archives (a bug?).
#   This genrule unpacks the caffe.a and merges the layers as a .o (ld -r).
#   (A terrible hack).
genrule(
    name = "caffe-extract",
    srcs = [":configure", "lib/libcaffe.a"],
    outs = ["libcaffe-layers.o"],
    cmd = '''
        workdir=$$(mktemp -d -t tmp.XXXXXXXXXX);
        cp $(location :lib/libcaffe.a) $$workdir;

        pushd $$workdir;
        ar x libcaffe.a;
        ld -r -o libcaffe-layers.o $$(echo layer_factory.cpp.o *_layer.*.o);
        popd;

        cp $$workdir/libcaffe-layers.o $(@D)/;
        rm -rf $$workdir;''',
)

cc_library(
    name = "openblas",
    srcs = ["lib/libopenblas.so.0"],
    data = ["lib/libopenblas.so.0"],
    hdrs = ["include/cblas.h", "include/openblas_config.h"],
    linkstatic = 1
)

cc_library(
    name = "lib",
    includes = ["include/"],
    srcs = [
        ":caffe-extract",
        "lib/libcaffe.a",
        "lib/libproto.a"
    ] + if_pycaffe([
        "lib/_caffe.cpp.o"
    ]),
    hdrs = glob(["include/**"]) + ["include/caffe/proto/caffe.pb.h"],
    defines = if_cuda(
        ["USE_CUDNN"],
        ["CPU_ONLY"]
    ) + if_pycaffe(
        ["WITH_PYTHON_LAYER"]
    ),
    deps = if_cuda([
        "@local_config_cuda//cuda:cudnn",
        "@local_config_cuda//cuda:cublas",
        "@local_config_cuda//cuda:curand"
    ]) + [
        "@protobuf//:protobuf",
        ":openblas",
    ],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu/hdf5/serial/lib",
        "-Wl,-rpath,/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial/lib",
        "-lboost_system",
        "-lboost_thread",
        "-lboost_filesystem",
        "-lpthread",
        "-lglog",
        "-lgflags",
        "-lhdf5_hl",
        "-lhdf5",
        "-lz",
        "-ldl",
        "-lm",
    ] + if_pycaffe([
        "-lpython2.7",
        "-lboost_python"
    ]),
    visibility = ["//visibility:public"],
    alwayslink = 1,
    linkstatic = 1,
)

