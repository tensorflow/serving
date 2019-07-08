# Release 1.12.3

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 1.12.3.

# Release 1.14.0

## Major Features and Improvements

* Use MKL-DNN contraction kernels by default. (commit: a075ebe5eff56f3311d6e2cc2d23e4e82567596b)
* Add option to refuse to unload the last servable version. (commit: c8496b199cedf3e38a7ad0dc4c46db2b341b28e5)
* Add ability to disable periodic filesystem polling (#1295). (commit: 72450555c83ea5e6d18d05362192ad85613b23b1)

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Add `enforce_session_run_timeout` inside `Server::Options`. (commit: de030640ec6ed2cd504ee0ad9335fb93aebe51b5)
* Add -o option, to pass params to `docker` command. (commit: dd59021d3f807f23390afa8a2bc34a6f7029ed24)
* Stop using reader locks (tf_shared_lock) on the read path of FastReadDynamicPtr. (commit: f04e583a6a700a4943a57b6758b3e131b0865e97)
* Add saved model tags to logging metadata. These tags are used by (commit: 6320701645d5aeceac49a4f02cc629159559f143)
* Adds an option in `SessionBundleConfig` to repeat warmup replay n times per request. (commit: 15cd20263c8362f534afecbdf98b9d929eac70fd)
* Improve tpu server warm up (commit: 63d31a33b4f6faeb0764bb159d403f2b49061aed)
* Official PIP package releases are now tied to a specific version of TensorFlow (commit: 9514c37d22f0b728e2db9e8c6f28fb11ebde0fad)
* Bump the minimal Bazel version to 0.24.1 (commit: 96a716ca31f753b0c3efc1ef60779b77f5c60845)
* Add new device type for TPU. (commit: c74861d61131e2248a70d9c72317df8c49eb8f1a)
* Fix incorrect formatting of decimal numbers in JSON output (#1332) (commit: d7c3b3deacbabf763ed44fb6932535016852e90a)
* Fixed the gzip uncompression support in the HTTP server for large request bodies. (commit: fb7835c7cd95c5b6b163cb2abd6a8b9a1a283689)
* Add stack memory resource kind. (commit: e56e72b3e4b9a597832734208a3da455f6db1a04)
* Adds ModelServer test for loading SavedModel exported from Keras Sequential API (commit: 9578f3d10c786c6714b9a8b481dd74f454402477)
* Ignore SIGPIPE for libevent，prevent the SIGPIPE signal from being raised (#1257) (commit: 8d88a5b3c4ac502113c798a470111ca65f47b0c2)
* Fix #1367 (commit: 58af9011d72cbd062501c3f8066bf4d9eee04a7a)
* Update Serving_REST_simple.ipynb (commit: 3870ba59a764d859fc137a8363588c94906e0f5f)
* Updates README with link to architecture overview (commit: d233a82e0a569d5ccd23a0cbada8099644698dc6)
* Update example section to use Docker (commit: a5fc8bbc20f712fd6c4c148ff4d94a9231b79ceb)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

G. Hussain Chinoy, Karthik Vadla, mkim301, yjhjstz

# Release 1.13.0

## Major Features and Improvements

* Support for TensorRT 5.0 (GPU docker image built against CUDA 10 and TensorRT 5.0)
* Support for listening gRPC over UNIX socket (commit: a25b0dad3984d3b154db1144df9d3b447b19aae6)
* New [GPU version of TensorFlow Serving API PIP package](https://pypi.org/project/tensorflow-serving-api-gpu/). This depends on the `tensorflow-gpu` instead of `tensorflow` PIP package, but is otherwise identical. (commit: 525c1af73ca543ce0165b3d22f0bbf21094fc443)
* [TF Serving end-to-end colab](tensorflow_serving/g3doc/tutorials/Serving_REST_simple.ipynb)! Training with Keras, serving with TF Serving and REST API (commit: 1ff8aadf20d75294aa4d496a807320603c6887c6)

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Make error message for input size mismatch in `Predict` call even more actionable. (commit: 7237fb54c8d5898713e0bba7573add60cd19c25e)
* Document how to use the version policy to pin a specific version, or serve multiple versions, of a model. (commit: 2724bfee911f1d2294a9ceb705bbd09a2701c344)
* Document config reloading and model version labels. (commit: f4890afdc42f10f125cba64c3c2f2c01309ba2e2)
* Fix the compile error on ARM-32 in net_http/server. (commit: 5446fd973de228693c1652acd4922dc4b177f77a)
* Adds ModelSpec to SessionRunResponse. (commit: 58a22637ef5e3c50153eb42eff652137eb18c94a)
* Add MKL support (commit: 8f792532bea10d82fd3c3b126412d0546f54ae28)
* Fix default path of Prometheus metrics endpoint (commit: 9d05b0c17be47d3260ab58c2b9ac97e202699b96)
* Add monitoring metrics for saved model (export_dir) warm up latency. (commit: de0935b64ec972879ae623aa4f438282a4281dcc)
* Add more details/clarification to model version labels documentation. (commit: f9e6ac4d60a4044fc3b8c07719d0faaeae401dda)
* Split `--tensorflow_session_parallelism` flag into two new flags: `--tensorflow_intra_op_parallelism` and `--tensorflow_inter_op_parallelism` (commit: 71092e448c5432f4411f7333a02b274f0a3cdd3f)
* Update CPU Docker images to Ubuntu 18.04 (commit: 8023fba48c5b47a81fec25c17ba385a720650ef8)
* Upgrade to Bazel 0.20.0 (commit: fc0b75f2e325a187794bf437ff3227510d261afb)
* Update Python 2 scripts to be compatible with both Python 2 and 3 (commit: 846d443bb506f07242cd99347901f3ad5b7efe6a)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Daniel Shi, Karthik Vadla, lapolonio, robert, Shintaro Murakami, Siju, Tom Forbes, Ville TöRhöNen

# Release 1.12.0

## Major Features and Improvements
* Add new REST API to [get model status](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md#model-status-api) from ModelServer (commit: 00e459f1604c40c073cbb9cb92d72cb6a88be9cd)
* Add new REST API to [get model metadata](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md#model-metadata-api) from ModelServer (fixes #1115) (commit: 97687024c3b7515d2f2979c35054f44c8f84d146)
* Support accepting gzipped REST API requests (fixes #1091) (commit: b94f6c89335782a7f175e8973c4f326375c55120)

## Breaking Changes

None

## Bug Fixes and Other Changes

* Update MKL build (commit: e11bd51540212242911dae00c8507e2852a5ad5a)
* Remove version pinning on pip packages (commit: 462072c2d78124c2769f820f7b63ee086de4e305)
* Update basic serving tutorials (commit: 33a4b052cedc39c21107bc99a090b59ca64ec568)
* Replacing legacy_init_op argument in SavedModelBuilder with main_op. (commit: 2fda31f905eefd2d108e9c84b8d7d55e4e482833)
* Add git hash for version metadata of model server and add tags for dev and nightly builds. (commit: 5c7740fc3d8d5c017643a8cc40a7202717b10dd6)
* Add error messages for specific cases when json for REST requests (commit: a17c89202e68bf19f369b9cbc97db7ced283b874)
* Python examples now run in a hermetic environment with all required dependencies (commit: 793fd90ee41ac34fa4c9261eef2d2c908dca9735)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Charles Verge, demfier, Kamidi Preetham, Lihang Li, naurril, vfdev, Yu Zheng

# Release 1.11.1

## Bug Fixes and Other Changes

* Fix version of model server binary (Fixes #1134)
* Range check floating point numbers correctly (Fixes #1136).
* Fix docker run script for same user and group name (Fixes #1137).
* Fix GPU build (Fixes #1150)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

vfdev

# Release 1.11.0

## Major Features and Improvements

* Prometheus exporter for TF metrics (see https://github.com/tensorflow/serving/commit/021efbd3281aa815cab0b35eab6d6d25249c12d4 for details).

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Built against TensorFlow [1.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.11.0)
* Accept integers for float/doubles in JSON REST API requests
* TF Serving API is now pre-built into Docker development images
* GPU Docker images are now built against cuDNN 7.2
* Add `--max_num_load_retries` flag to ModelServer (fixes #1099)
* Add user-configured model version labels to the stand-alone ModelServer binary.
* Directly import tensor.proto.h (the transitive import will be removed from tensor.h soon)
* Building optimized TensorFlow Serving binaries is now easier (see [docs](https://github.com/tensorflow/serving/g3doc/setup.md]) for details)
* Adds columnar format support for input/output tensors in Predict REST API (fixes #1047)
* Development Dockerfiles now produce a more optimized ModelServer
* Fixed TensorFlow Serving API PyPi package overwriting TensorFlow package.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Feisan, joshua.horowitz, Prashanth Reddy Basani, tianyapiaozi, Vamsi Sripathi, Yu Zheng

# Release 1.11.0-rc0

## Major Features and Improvements

* Prometheus exporter for TF metrics (see https://github.com/tensorflow/serving/commit/021efbd3281aa815cab0b35eab6d6d25249c12d4 for details).
* Added new REST API to [get status of model(s)](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md#model-status-api) from ModelServer.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Built against TensorFlow [1.11.0-rc0](https://github.com/tensorflow/tensorflow/releases/tag/v1.11.0-rc0).
* Directly import tensor.proto.h (the transitive import will be removed from tensor.h soon)
* Building optimized TensorFlow Serving binaries is now easier (see [docs](https://github.com/tensorflow/serving/g3doc/setup.md]) for details)
* Adds columnar format support for input/output tensors in Predict REST API (fixes #1047)
* Development Dockerfiles now produce a more optimized ModelServer
* Fixed TensorFlow Serving API PyPi package overwriting TensorFlow package.

# Release 1.10.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* TensorFlow Serving API now uses gRPC's GA release. The beta gRPC API has been deprecated, and will be removed in a future version of TensorFlow Serving. Please update your gRPC client code ([sample](https://github.com/tensorflow/serving/commit/aa35cfdb24016f6d88f82c53d45c8ce9fa550499#diff-e7d756a12c65a8b5ac90229b23523023))
* Docker images for GPU are built against NCCL 2.2, in following with Tensorflow 1.10.

## Bug Fixes and Other Changes

* Built against TensorFlow 1.10.
* Added GPU serving Docker image.
* Repo cloning and shell prompt in example readme.
* Updated Docker instructions.
* Updated min Bazel version (0.15.0).
* Convert TF_CHECK_OKs to TF_ASSERT_OK in some unit tests.
* Remove error suppression (.IgnoreError()) from BasicManager.
* Add new bazel_in_docker.sh tool for doing hermetic bazel builds.
* Fix erroneous formatting of numbers in REST API output that are larger than 6 digits.
* Add support for Python 3 while also compatible with Python 2.7 in mnist_saved_model.py.
* Fix an incorrect link to Dockerfile.devel-gpu.
* Add util for get model status.
* Adding support for secure channel to ModelServer.
* Add version output to model server binary.
* Change ServerRequestLogger::Update to only create new and delete old loggers if needed.
* Have the Model Server interpret specific hard-coded model version labels "stable" and "canary" as the smallest and largest version#, respectively.
* Add half_plus_two CPU and GPU models to test data.

# Release 0.4.0

Initial release of TensorFlow Serving.
