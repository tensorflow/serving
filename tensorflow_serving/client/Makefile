CC= g++
CFLAGS= -g -std=c++11 -static

ROOT_PATH=.
TARGET = libtfclient.a
INC_DIR= -I.
LIB_DIR=/usr/local/lib/
LIBS_SERVER= -Bstatic -libprotobuf.a -libgrpc.a -libgrpc++.a,-Bstatic -Wl,-Bdynamic
SOUCES=tf_client.cc prediction_service.grpc.pb.cc prediction_service.pb.cc predict.grpc.pb.cc resource_handle.pb.cc tensor.pb.cc predict.pb.cc model.pb.cc tensor_shape.pb.cc types.pb.cc
OBJ = $(SOURCES:.cc=.o)
all: $(TARGET)
$(TARGET): $(OBJ)
    $(CC) $(CFLAGS) -o $(TARGET) $(OBJ) ${INC_DIR}
