# How to Configure a Model Server

Create a file containing an ASCII
[ModelServerConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/model_server_config.proto#L76)
protocol buffer, and pass its path to the server using the --model_config_file
flag. (Some useful references:
[what an ASCII protocol buffer looks like](https://stackoverflow.com/questions/18873924/what-does-the-protobuf-text-format-look-like);
[how to pass flags in Docker](docker.md#passing-additional-arguments).)

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
  }
  config {
    name: 'my_second_model'
    base_path: '/tmp/my_second_model/'
  }
}
```

## Configuring One Model

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

