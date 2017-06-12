# This Makefile compiles a library containing the C++ runtime for the TensorFlow
# library. It's designed for use on platforms with limited resources where
# running a full Bazel build would be prohibitive, or for cross-compilation onto
# embedded systems. It includes only a bare-bones set of functionality.
#
# The default setup below is aimed at Unix-like devices, and should work on
# modern Linux and OS X distributions without changes.
#
# If you have another platform, you'll need to take a careful look at the
# compiler flags and folders defined below. They're separated into two sections,
# the first for the host (the machine you're compiling on) and the second for
# the target (the machine you want the program to run on).

# Host compilation settings

MAKEFILE_DIR := .
HAS_GEN_HOST_PROTOC := \
$(shell test -f $(MAKEFILE_DIR)/gen/grpc/bin/protoc && echo "true" ||\
echo "false")

SOURCE_SLICE_DIR := $(MAKEFILE_DIR)/code-slice/
PROTOC_INCLUDE_DIRS := $(SOURCE_SLICE_DIR)

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = WINDOWS
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
	        HOST_OS := LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := OSX
	endif
endif

# Where compiled objects are stored.
HOST_OBJDIR := $(MAKEFILE_DIR)/gen/host_obj/
HOST_BINDIR := $(MAKEFILE_DIR)/gen/host_bin/
HOST_GENDIR := $(MAKEFILE_DIR)/gen/host_obj/

# Settings for the host compiler.
HOST_CXX := $(CC_PREFIX) gcc
HOST_CXXFLAGS := --std=c++11
HOST_LDOPTS :=
ifeq ($(HAS_GEN_HOST_PROTOC),true)
	HOST_LDOPTS += -L$(MAKEFILE_DIR)/gen/grpc/lib
endif
HOST_LDOPTS += -L/usr/local/lib

HOST_INCLUDES := \
-I$(SOURCE_SLICE_DIR) \
-I$(MAKEFILE_DIR)/downloads \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(HOST_GENDIR) 

ifeq ($(HAS_GEN_HOST_PROTOC),true)
	HOST_INCLUDES += -I$(MAKEFILE_DIR)/gen/grpc/include
endif
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
HOST_INCLUDES += -I/usr/local/include

HOST_STATIC_LIBS := \
gen/grpc/lib/libprotobuf.a 

HOST_LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz

# If we're on Linux, also link in the dl library.
ifeq ($(HOST_OS),LINUX)
	HOST_LIBS += -ldl -lpthread
endif

# proto_text is a tool that converts protobufs into a form we can use more
# compactly within TensorFlow. It's a bit like protoc, but is designed to
# produce a much more minimal result so we can save binary space.
# We have to build it on the host system first so that we can create files
# that are needed for the runtime building.
PROTO_TEXT := $(HOST_BINDIR)proto_text

# The list of dependencies is derived from the Bazel build file by running
# the gen_file_lists.sh script on a system with a working Bazel setup.
PROTO_TEXT_CC_FILES := $(shell cat $(MAKEFILE_DIR)/proto_text_util_cc_files.txt)
PROTO_TEXT_PB_CC_LIST := $(shell cat $(MAKEFILE_DIR)/proto_text_util_pb_cc_files.txt)
PROTO_TEXT_PB_H_LIST := $(shell cat $(MAKEFILE_DIR)/proto_text_util_pb_h_files.txt)

PROTO_TEXT_CC_FILES_NORMALIZED := $(subst tensorflow/tensorflow/,tensorflow/, $(PROTO_TEXT_CC_FILES))
PROTO_TEXT_PB_CC_LIST_NORMALIZED := $(subst tensorflow/tensorflow/,tensorflow/, $(PROTO_TEXT_PB_CC_LIST))
PROTO_TEXT_PB_H_LIST_NORMALIZED := $(subst tensorflow/tensorflow/,tensorflow/, $(PROTO_TEXT_PB_H_LIST))

# Locations of the intermediate files proto_text generates.
PROTO_TEXT_PB_H_FILES := $(addprefix $(HOST_GENDIR), $(PROTO_TEXT_PB_H_LIST_NORMALIZED))
PROTO_TEXT_CC_OBJS := $(addprefix $(HOST_OBJDIR), $(PROTO_TEXT_CC_FILES_NORMALIZED:.cc=.o))
PROTO_TEXT_PB_OBJS := $(addprefix $(HOST_OBJDIR), $(PROTO_TEXT_PB_CC_NORMALIZED:.cc=.o))
PROTO_TEXT_OBJS := $(PROTO_TEXT_CC_OBJS) $(PROTO_TEXT_PB_OBJS)

# Target device settings.

# Default to running on the same system we're compiling on.
# You should override TARGET on the command line if you're cross-compiling, e.g.
# make -f tensorflow/contrib/makefile/Makefile TARGET=ANDROID
TARGET := $(HOST_OS)

# Where compiled objects are stored.
GENDIR := $(MAKEFILE_DIR)/gen/
OBJDIR := $(GENDIR)obj/
LIBDIR := $(GENDIR)lib/
BINDIR := $(GENDIR)bin/
PBTGENDIR := $(GENDIR)proto_text/
PROTOGENDIR := $(GENDIR)proto_pb/
DEPDIR := $(GENDIR)dep/
$(shell mkdir -p $(DEPDIR) >/dev/null)

# Settings for the target compiler.
CXX := $(CC_PREFIX) gcc
OPTFLAGS := -O2
OPTFLAGS += -march=native

CXXFLAGS := --std=c++11 -fno-exceptions -DNDEBUG $(OPTFLAGS)
LDFLAGS := \
-L/usr/local/lib

DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

INCLUDES := \
-I$(SOURCE_SLICE_DIR) \
-I$(MAKEFILE_DIR)/downloads \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(PROTOGENDIR) \
-I$(PBTGENDIR)
ifeq ($(HAS_GEN_HOST_PROTOC),true)
	INCLUDES += -I$(MAKEFILE_DIR)/gen/grpc/include
endif
# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/local/include

# TODO(mtm): Add TARGET OS specific *LIBS for the Mac.
# Possibly a static link to: downloads/grpc/libs/opt/libboringssl.a

STATIC_LIBS := \
gen/grpc/lib/libgrpc++.a \
gen/grpc/lib/libgrpc.a \
gen/grpc/lib/libgrpc++_reflection.a \
gen/grpc/lib/libgrpc++_unsecure.a \
gen/grpc/lib/libgrpc_unsecure.a \
gen/grpc/lib/libgpr.a \
gen/grpc/lib/libprotobuf.a 

LIBS := \
-lstdc++ \
-lpng \
-lz \
-lssl -lcrypto \
-lm

ifeq ($(HAS_GEN_HOST_PROTOC),true)
	PROTOC := $(MAKEFILE_DIR)/gen/grpc/bin/protoc
	PROTOC_GRPC_PLUGIN := $(MAKEFILE_DIR)/gen/grpc/bin/grpc_cpp_plugin
else
	PROTOC := protoc
	PROTOC_GRPC_PLUGIN := grpc_cpp_plugin
endif

$(info PROTOC = "$(PROTOC)")
$(info CC_PREFIX = "$(CC_PREFIX)")

PROTOCFLAGS := -I $(PROTOC_INCLUDE_DIRS)
AR := ar
ARFLAGS := -r
LIBFLAGS :=

# If we're on OS X, make sure that globals aren't stripped out.
ifeq ($(TARGET),OSX)
	LDFLAGS += -all_load
endif
# Make sure that we don't strip global constructors on Linux.
ifeq ($(TARGET),LINUX)
ifeq ($(HAS_GEN_HOST_PROTOC),true)
	LIBFLAGS += -L$(MAKEFILE_DIR)/gen/grpc/lib
	export LD_LIBRARY_PATH=$(MAKEFILE_DIR)/gen/grpc/lib
endif
	CXXFLAGS += -fPIC
	LIBFLAGS += -Wl,--allow-multiple-definition -Wl,--whole-archive
	LDFLAGS := -Wl,--no-whole-archive
endif

# If we're on Linux, also link in the dl library.
ifeq ($(TARGET),LINUX)
	LIBS += -ldl -lpthread
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-serving-client.a
LIB_PATH := $(LIBDIR)$(LIB_NAME)

# A small example program that shows how to link against the library.
INCEPTION_CLIENT := $(BINDIR)inception_client

# The example client that will be compiled using the generated library
INCEPTION_CLIENT_SRCS := \
tensorflow_serving/example/inception_client.cc

# We don't want the .o file of the example client to be put into the library
CLIENT_MAIN_EXCLUDE_SRCS := $(INCEPTION_CLIENT_SRCS)

# Create lists for this makefile to work on, from generated file lists.
# Add any additional files needed here at the begining.
TF_CC_SRCS := tensorflow/core/util/version_info.cc
TF_CC_SRCS += $(shell cat $(MAKEFILE_DIR)/cc_client_cc_source_files.txt)
PBT_CC_SRCS := $(shell cat $(MAKEFILE_DIR)/cc_client_pb_text_cc_generated_files.txt)
PROTO_SRCS := $(shell cat $(MAKEFILE_DIR)/cc_client_proto_source_files.txt)
PROTO_GRPC_SRCS := $(shell cat $(MAKEFILE_DIR)/cc_client_grpc_proto_generated_files.txt)
TF_CC_SRCS_NORMALIZED_ALL := $(subst tensorflow/tensorflow/,tensorflow/, $(TF_CC_SRCS))
TF_CC_SRCS_NORMALIZED := $(filter-out $(CLIENT_MAIN_EXCLUDE_SRCS), $(TF_CC_SRCS_NORMALIZED_ALL))
PBT_CC_SRCS_NORMALIZED := $(subst tensorflow/tensorflow/,tensorflow/, $(PBT_CC_SRCS))
PROTO_SRCS_NORMALIZED := $(subst tensorflow/tensorflow/,tensorflow/, $(PROTO_SRCS))
PROTO_GRPC_SRCS_NORMALIZED := $(subst tensorflow/tensorflow/,tensorflow/, $(PROTO_GRPC_SRCS))

# File names of the intermediate files target compilation generates.
TF_CC_OBJS := $(addprefix $(OBJDIR), $(TF_CC_SRCS_NORMALIZED:.cc=.o))
PBT_GEN_FILES := $(addprefix $(PBTGENDIR), $(PBT_CC_SRCS_NORMALIZED))
PBT_OBJS := $(addprefix $(OBJDIR), $(PBT_CC_SRCS_NORMALIZED:.cc=.o))
PROTO_CC_SRCS := $(addprefix $(PROTOGENDIR), $(PROTO_SRCS_NORMALIZED:.proto=.pb.cc))
PROTO_GRPC_CC_SRCS := $(addprefix $(PROTOGENDIR), $(PROTO_GRPC_SRCS_NORMALIZED:.proto=.grpc.pb.cc))
PROTO_OBJS := $(addprefix $(OBJDIR), $(PROTO_SRCS_NORMALIZED:.proto=.pb.o))
PROTO_GRPC_OBJS := $(addprefix $(OBJDIR), $(PROTO_GRPC_SRCS_NORMALIZED:.proto=.grpc.pb.o))
LIB_OBJS := $(PROTO_OBJS) $(TF_CC_OBJS) $(PBT_OBJS)
BENCHMARK_OBJS := $(addprefix $(OBJDIR), $(BENCHMARK_SRCS:.cc=.o))
INCEPTION_CLIENT_OBJS := $(addprefix $(OBJDIR), $(INCEPTION_CLIENT_SRCS:.cc=.o))

.PHONY: clean cleantarget clean_except_protobuf_libs

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH) $(INCEPTION_CLIENT)

# Rules for target compilation.

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): $(LIB_OBJS) $(PROTO_GRPC_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(PROTO_GRPC_OBJS) $(LIB_OBJS) 

$(INCEPTION_CLIENT): $(INCEPTION_CLIENT_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(INCEPTION_CLIENT) $(INCEPTION_CLIENT_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(STATIC_LIBS) $(LDFLAGS) $(LIBS)

# Matches on the normal hand-written TensorFlow C++ source files.
$(OBJDIR)%.o: $(SOURCE_SLICE_DIR)%.cc | $(PBT_GEN_FILES)
	@mkdir -p $(dir $@)
	@mkdir -p $(dir $(DEPDIR)$*)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@
	@mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

# Compiles C++ source files that have been generated by protoc.
$(OBJDIR)%.pb.o: $(PROTOGENDIR)%.pb.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compiles grpc C++ source files that have been generated by protoc.
$(OBJDIR)%.grpc.pb.o: $(PROTOGENDIR)%.grpc.pb.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Builds C++ code from proto files using protoc.
$(PROTOGENDIR)%.pb.cc $(PROTOGENDIR)%.pb.h: $(SOURCE_SLICE_DIR)%.proto
	@mkdir -p $(dir $@)
	$(PROTOC) $(PROTOCFLAGS) $< --cpp_out $(PROTOGENDIR)

# Generates the grpc.pb  C++ source files from protos using protoc.
$(PROTOGENDIR)%.grpc.pb.cc $(PROTOGENDIR)%.grpc.pb.h: $(SOURCE_SLICE_DIR)%.proto
	@mkdir -p $(dir $@)
	$(PROTOC) $(PROTOCFLAGS) $< --grpc_out $(PROTOGENDIR) --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN)

# Uses proto_text to generate minimal pb_text C++ files from protos.
$(PBTGENDIR)%.pb_text.cc $(PBTGENDIR)%.pb_text.h $(PBTGENDIR)%.pb_text-impl.h: $(SOURCE_SLICE_DIR)%.proto | $(PROTO_TEXT)
# Note: proto_text only seems to work correctly if run from root of source code dir. 
# So hacking paths here to accommodate this.
	@mkdir -p $(dir $@)
	cd $(SOURCE_SLICE_DIR) && \
	../gen/host_bin/proto_text \
	../gen/proto_text/tensorflow/core \
	tensorflow/core/ \
	tensorflow/tools/proto_text/placeholder.txt \
        $(subst code-slice/,, $<)

# Compiles the C++ source files created by proto_text.
$(OBJDIR)%.pb_text.o: $(PBTGENDIR)%.pb_text.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Makes sure that we don't compile the protoc-generated C++ sources before they
# and the proto_text files have been created.
$(PROTO_OBJS): $(PROTO_CC_SRCS) $(PROTO_GRPC_CC_SRCS) $(PBT_GEN_FILES)

# Host compilation rules.

# For normal manually-created TensorFlow C++ source files.
$(HOST_OBJDIR)%.o: $(SOURCE_SLICE_DIR)%.cc
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_INCLUDES) -c $< -o $@

# Compiles object code from protoc-built C++ source files.
$(HOST_OBJDIR)%.pb.o: $(HOST_GENDIR)%.pb.cc
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_INCLUDES) -c $< -o $@

# Compiles object code from grpc protoc-built C++ source files.
$(HOST_OBJDIR)%.grpc.pb.o: $(HOST_GENDIR)%.grpc.pb.cc
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_INCLUDES) -c $< -o $@

# Ensures we wait until proto_text has generated the .h files from protos before
# we compile the C++.
$(PROTO_TEXT_OBJS) : $(PROTO_TEXT_PB_H_FILES)

# Compiles the proto_text utility, that will be used to generate proto_text C++ source files from protos.
$(PROTO_TEXT): $(PROTO_TEXT_OBJS) $(PROTO_TEXT_PB_H_FILES)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(HOST_CXXFLAGS) $(HOST_INCLUDES) \
	-o $(PROTO_TEXT) $(PROTO_TEXT_OBJS) $(HOST_LDOPTS) $(HOST_STATIC_LIBS)  $(HOST_LIBS)

# Generates the pb C++ source files from protos using protoc.
$(HOST_GENDIR)%.pb.cc $(HOST_GENDIR)%.pb.h: $(SOURCE_SLICE_DIR)%.proto
	@mkdir -p $(dir $@)
	$(PROTOC) $(PROTOCFLAGS) $< --cpp_out $(HOST_GENDIR)

# Generates the grpc.pb  C++ source files from protos using protoc.
$(HOST_GENDIR)%.grpc.pb.cc $(HOST_GENDIR)%.grpc.pb.h: $(SOURCE_SLICE_DIR)%.proto
	@mkdir -p $(dir $@)
	$(PROTOC) $(PROTOCFLAGS) $< --grpc_out $(HOST_GENDIR) --plugin=protoc-gen-grpc=$(PROTOC_GRPC_PLUGIN)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Gets rid of all generated files except protobuf/grpc libs generated
# before calling make.  This allows users not to recompile proto libs everytime.
clean_except_protobuf_libs:
	find $(MAKEFILE_DIR)/gen -mindepth 1 -maxdepth 1 ! -name "grpc" -exec rm -r "{}" \;

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(TF_CC_SRCS_NORMALIZED)))

