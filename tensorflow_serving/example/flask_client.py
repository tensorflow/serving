import flask
import numpy as np

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

app = flask.Flask(__name__)


@app.route('/model_prediction', methods=["GET", "POST"])
def model_prediction():
    host = "localhost"
    port = 9000
    model_name = "default"
    input_name = flask.request.values.get('input_name')
    input_type = flask.request.values.get('input_type')
    input = np.array(flask.request.values.getlist('input'), dtype=input_type)


    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name

    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(input))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    return str(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888)
