# Tensorflow Serving Configuration

In this guide, we will go over the numerous configuration points for Tensorflow
Serving.

## Overview

While most configurations relate to the Model Server, there are many ways to
specify the behavior of Tensorflow Serving:

*   [Model Server Configuration](#model-server-configuration): Specify model
    names, paths, version policy & labels, logging configuration and more
*   [Monitoring Configuration](#monitoring-configuration): Enable and configure
    Prometheus monitoring
*   [Batching Configuration](#batching-configuration): Enable batching and
    configure its parameters
*   [Misc. Flags](#miscellaneous-flags): A number of misc. flags that can be
    provided to fine-tune the behavior of a Tensorflow Serving deployment

## Model Server Configuration

The easiest way to serve a model is to provide the `--model_name` and
`--model_base_path` flags (or setting the `MODEL_NAME` environment variable if
using Docker). However, if you would like to serve multiple models, or configure
options like polling frequency for new versions, you may do so by writing a
Model Server config file.

You may provide this configuration file using the `--model_config_file` flag and
instruct Tensorflow Serving to periodically poll for updated versions of this
configuration file at the specifed path by setting the
`--model_config_file_poll_wait_seconds` flag.

Example using Docker:

```
docker run -t --rm -p 8501:8501 \
    -v "$(pwd)/models/:/models/" tensorflow/serving \
    --model_config_file=/models/models.config \
    --model_config_file_poll_wait_seconds=60
```

### Reloading Model Server Configuration

There are two ways to reload the Model Server configuration:

*   By setting the `--model_config_file_poll_wait_seconds` flag to instruct the
    server to periodically check for a new config file at `--model_config_file`
    filepath.

*   By issuing
    [HandleReloadConfigRequest](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/model_service.proto#L22)
    RPC calls to the server and supplying a new Model Server config
    programmatically.

Please note that each time the server loads the new config file, it will act to
realize the content of the new specified config and _only_ the new specified
config. This means if model A was present in the first config file, which is
replaced with a file that contains only model B, the server will load model B
and unload model A.

### Model Server Config Details

The Model Server configuration file provided must be a
[ModelServerConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/model_server_config.proto#L76)
[protocol buffer](https://stackoverflow.com/questions/18873924/what-does-the-protobuf-text-format-look-like).

For all but the most advanced use-cases, you'll want to use the ModelConfigList
option, which is a list of
[ModelConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/model_server_config.proto#L19)
protocol buffers. Here's a basic example, before we dive into advanced options
below.

```proto
model_config_list {
  config {
    name: 'my_first_model'
    base_path: '/tmp/my_first_model/'
    model_platform: 'tensorflow'
  }
  config {
    name: 'my_second_model'
    base_path: '/tmp/my_second_model/'
    model_platform: 'tensorflow'
  }
}
```

### Configuring One Model

Each ModelConfig specifies one model to be served, including its name and the
path where the Model Server should look for versions of the model to serve, as
seen in the above example. By default the server will serve the version with the
largest version number. This default can be overridden by changing the
model_version_policy field.

### Serving a Specific Version of a Model

To serve a specific version of the model, rather than always transitioning to
the one with the largest version number, set model_version_policy to "specific"
and provide the version number you would like to serve. For example, to pin
version 42 as the one to serve:

```proto
model_version_policy {
  specific {
    versions: 42
  }
}
```

This option is useful for rolling back to a know good version, in the event a
problem is discovered with the latest version(s).

### Serving Multiple Versions of a Model

To serve multiple versions of the model simultaneously, e.g. to enable canarying
a tentative new version with a slice of traffic, set model_version_policy to
"specific" and provide multiple version numbers. For example, to serve versions
42 and 43:

```proto
model_version_policy {
  specific {
    versions: 42
    versions: 43
  }
}
```

### Assigning String Labels to Model Versions, To Simplify Canary and Rollback

Sometimes it's helpful to add a level of indirection to model versions. Instead
of letting all of your clients know that they should be querying version 42, you
can assign an alias such as "stable" to whichever version is currently the one
clients should query. If you want to redirect a slice of traffic to a tentative
canary model version, you can use a second alias "canary".

You can configure these model version aliases, or labels, like so:

```proto
model_version_policy {
  specific {
    versions: 42
    versions: 43
  }
}
version_labels {
  key: 'stable'
  value: 42
}
version_labels {
  key: 'canary'
  value: 43
}
```

In the above example, you are serving versions 42 and 43, and associating the
label "stable" with version 42 and the label "canary" with version 43. You can
have your clients direct queries to one of "stable" or "canary" (perhaps based
on hashing the user id) using the version_label field of the
[ModelSpec](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/model.proto#L27)
protocol buffer, and move forward the label on the server without notifying the
clients. Once you are done canarying version 43 and are ready to promote it to
stable, you can update the config to:

```proto
model_version_policy {
  specific {
    versions: 42
    versions: 43
  }
}
version_labels {
  key: 'stable'
  value: 43
}
version_labels {
  key: 'canary'
  value: 43
}
```

If you subsequently need to perform a rollback, you can revert to the old config
that has version 42 as "stable". Otherwise, you can march forward by unloading
version 42 and loading the new version 44 when it is ready, and then advancing
the canary label to 44, and so on.

Please note that labels can only be assigned to model versions that are
_already_ loaded and available for serving. Once a model version is available,
one may reload the model config on the fly to assign a label to it. This can be
achieved using a
[HandleReloadConfigRequest](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/model_service.proto#L22)
RPC or if the server is set up to periodically poll the filesystem for the
config file, as described [above](#reloading-model-server-configuration).

If you would like to assign a label to a version that is not yet loaded (for ex.
by supplying both the model version and the label at startup time) then you must
set the `--allow_version_labels_for_unavailable_models` flag to true, which
allows new labels to be assigned to model versions that are not loaded yet.

Please note that this applies only to new version labels (i.e. ones not assigned
to a version currently). This is to ensure that during version swaps, the server
does not prematurely assign the label to the new version, thereby dropping all
requests destined for that label while the new version is loading.

In order to comply with this safety check, if re-assigning an already in-use
version label, you must assign it only to already-loaded versions. For example,
if you would like to move a label from pointing to version N to version N+1, you
may first submit a config containing version N and N+1, and then submit a config
that contains version N+1, the label pointing to N+1 and no version N.

#### REST Usage

If you're using the REST API surface to make inference requests, instead of
using

`/v1/models/<model name>/versions/<version number>`

simply request a version using a label by structuring your request path like so

`/v1/models/<model name>/labels/<version label>`.

Note that version label is restricted to a sequence of Word characters, composed
of alphanumeral characters and underscores (i.e. `[a-zA-Z0-9_]+`).

## Monitoring Configuration

You may provide a monitoring configuration to the server by using the
`--monitoring_config_file` flag to specify a file containing a
[MonitoringConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/monitoring_config.proto#L17)
protocol buffer. Here's an example:

```proto
prometheus_config {
  enable: true,
  path: "/monitoring/prometheus/metrics"
}
```

To read metrics from the above monitoring URL, you first need to enable the HTTP
server by setting the `--rest_api_port` flag. You can then configure your
Prometheus Server to pull metrics from Model Server by passing it the values of
`--rest_api_port` and `path`.

Tensorflow Serving collects all metrics that are captured by Serving as well as
core Tensorflow.

## Batching Configuration

Model Server has the ability to batch requests in a variety of settings in order
to realize better throughput. The scheduling for this batching is done globally
for all models and model versions on the server to ensure the best possible
utilization of the underlying resources no matter how many models or model
versions are currently being served by the server
([more details](http://github.com/tensorflow/serving/tree/master/tensorflow_serving/batching/README.md#servers-with-multiple-models-model-versions-or-subtasks)).
You may enable this behavior by setting the `--enable_batching` flag and control
it by passing a config to the `--batching_parameters_file` flag.

Example batching parameters file:

```
max_batch_size { value: 128 }
batch_timeout_micros { value: 0 }
max_enqueued_batches { value: 1000000 }
num_batch_threads { value: 8 }
```

Please refer to the
[batching guide](http://github.com/tensorflow/serving/tree/master/tensorflow_serving/batching/README.md)
for an in-depth discussion and refer to the
[section on parameters](http://github.com/tensorflow/serving/tree/master/tensorflow_serving/batching/README.md#batch-scheduling-parameters-and-tuning)
to understand how to set the parameters.

## Miscellaneous Flags

In addition to the flags covered so far in the guide, here we list a few other
notable ones. For a complete list, please refer to the
[source code](http://github.com/tensorflow/serving/tree/master/tensorflow_serving/model_servers/main.cc#L59).

*   `--port`: Port to listen on for gRPC API
*   `--rest_api_port`: Port to listen on for HTTP/REST API
*   `--rest_api_timeout_in_ms`: Timeout for HTTP/REST API calls
*   `--file_system_poll_wait_seconds`: The period with which the server polls
    the filesystem for new model versions at each model's respective
    model_base_path
*   `--enable_model_warmup`: Enables [model warmup](saved_model_warmup.md) using
    user-provided PredictionLogs in assets.extra/ directory
