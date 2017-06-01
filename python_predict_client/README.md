# Python Predict Client

It is the independent Python predict client for TensorFlow Serving.

## Installation

Copy the proto files.

```
cp ../tensorflow_serving/apis/{model,predict,prediction_service}.proto .
```

Replce the import path.

```
sed -i "s/tensorflow_serving\/apis\///g" {predict,prediction_service}.proto
```

Comppile the proto files.

```
python -m grpc.tools.protoc --proto_path=../tensorflow --proto_path=. --python_out=. --grpc_python_out=. {model,predict,prediction_service}.proto
```

## Inception Client

```
python ./inception_client.py --server=127.0.0.1:9000 --image=foo.jpg
```
