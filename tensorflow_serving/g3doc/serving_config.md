# How to Configure a Model Server

Create a file containing an ASCII
[ModelServerConfig](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/config/model_server_config.proto#L76)
protocol buffer, and pass its path to the server using the --model_config_file
flag. (Some useful references:
[what an ASCII protocol buffer looks like](https://stackoverflow.com/questions/18873924/what-does-the-protobuf-text-format-look-like);
[how to pass flags in Docker](docker.md#passing-additional-arguments).)

You can also reload the model config on the fly, after the server is running,
via the
[HandleReloadConfigRequest](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/model_service.proto#L22)
RPC endpoint. This will cause models in the new config that are not in the old
config to be loaded, and models in the old config that are not in the new config
to be unloaded; (models in both configs will remain in place, and will not be
transiently unloaded).

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

Please note that labels can only be assigned to model versions that are loaded
and available for serving. Once a model version is available, one may reload
the model config on the fly, to assign a label to it
(can be achieved using
[HandleReloadConfigRequest](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/model_service.proto#L22)
RPC endpoint).
