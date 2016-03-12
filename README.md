#TensorFlow Serving

[![Build Status](http://ci.tensorflow.org/buildStatus/icon?job=serving-master-cpu)](http://ci.tensorflow.org/job/serving-master-cpu)

TensorFlow Serving is an open-source software library for serving
machine learning models. It deals with the *inference* aspect of machine
learning, taking models after *training* and managing their lifetimes, providing
clients with versioned access via a high-performance, reference-counted lookup
table.

Multiple models, or indeed multiple versions of the same model, can be served
simultaneously. This flexibility facilitates canarying new versions,
non-atomically migrating clients to new models or versions, and A/B testing
experimental models.

The primary use-case is high-performance production serving, but the same
serving infrastructure can also be used in bulk-processing (e.g. map-reduce)
jobs to pre-compute inference results or analyze model performance. In both
scenarios, GPUs can substantially increase inference throughput. TensorFlow
Serving comes with a scheduler that groups individual inference requests into
batches for joint execution on a GPU, with configurable latency controls.

TensorFlow Serving has out-of-the-box support for TensorFlow models (naturally),
but at its core it manages arbitrary versioned items (*servables*) with
pass-through to their native APIs. In addition to trained TensorFlow models,
servables can include other assets needed for inference such as embeddings,
vocabularies and feature transformation configs, or even non-TensorFlow-based
machine learning models.

The architecture is highly modular. You can use some parts individually (e.g.
batch scheduling) or use all the parts together. There are numerous plug-in
points; perhaps the most useful ways to extend the system are:
(a) [creating a new type of servable](tensorflow_serving/g3doc/custom_servable.md);
(b) [creating a custom source of servable versions](tensorflow_serving/g3doc/custom_source.md).

**If you'd like to contribute to TensorFlow Serving, be sure to review the
[contribution guidelines](CONTRIBUTING.md).**

**We use [GitHub issues](https://github.com/tensorflow/serving/issues) for
tracking requests and bugs.

# Download and Setup

See [install instructions](tensorflow_serving/g3doc/setup.md).

##Tutorials

* [Basic tutorial](tensorflow_serving/g3doc/serving_basic.md)
* [Advanced tutorial](tensorflow_serving/g3doc/serving_advanced.md)

##For more information

* [Serving architecture overview](tensorflow_serving/g3doc/architecture_overview.md)
* [TensorFlow website](http://tensorflow.org)
