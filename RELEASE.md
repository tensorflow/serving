# Release 2.1.0

## Major Features and Improvements
* Add integration with [TensorBoard profiler service](https://www.tensorflow.org/tensorboard).

## Breaking Changes

## Bug Fixes and Other Changes

* Fix link for TFRecord in Saved Model Warmup documentation. (commit: 127a112a91bda3d7d3c3a56802632376bbe3e36e)
* Fix typo in http server log message. (commit: 509f6da062dc9b091ad6961a94740cf64e265c36)
* Be able to discard aspired-versions request from SourceRouter (commit: 10e4987502ee91fe74c6c179ed4ba52f17cc75b4)
* Use public tf.config APIs (commit: 87a4b2b28729bd269ab367742998b6f8426ea1b7)
* Fix copying of string tensor outputs by explicitly copying each (commit: 9544077bdb6eef9b20a0688a042155ee6dea011a)
* Migrate from std::string to tensorflow::tstring. (commit: e24571ac9ce390733f3b02188c7d740f08fff62d)

# Release 2.0.0

## Major Features and Improvements
* Some Tensorflow Text ops have been added to ModelServer (specifically constrained_sequence_op, sentence_breaking_ops, unicode_script_tokenizer, whitespace_tokenizer, wordpiece_tokenizer)

## Breaking Changes
* As previously announced[1](https://groups.google.com/a/tensorflow.org/forum/#!msg/announce/qXfsxr2sF-0/jHQ77dr3DAAJ)[2](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)[3](https://github.com/tensorflow/serving/releases/tag/1.15.0), Contrib ops will not be packaged with Tensorflow, and therefore will not be available in Tensorflow Serving. If serving with Tensorflow Serving >1.15, please ensure your models do not contain any tf.contrib ops. If you are critically dependent on custom ops, please review [this guide](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/custom_op.md) for instructions to statically build ops into the model server.
* After being [deprecated](https://developers.googleblog.com/2017/07/tensorflow-serving-10.html) for multiple years, as a part of tf.contrib deprecation, SessionBundle API will be removed starting from Tensorflow Serving 2.0 - if currently using SessionBundle, please migrate to SavedModel APIs.

## Bug Fixes and Other Changes
* Add a section in the documentation for testing custom op manually. (commit: 1b65af1d7fee4fe79b4152f94d5ea422e8a79cca)
* Add ops delegate library to enable running TF ops. (commit: 14112359d16b3e1e275c2ba70b0e078ce4863783)
* Add command line tool to load TF Lite model for manual testing/debugging. (commit: 0b0254d4a90550b1d7228334187e624bf4b31c37)
* Fixes broken relative docs links (commit: 12813143b22616091388e7659d7f69cfcf518269)
* Cleaning up BUILD visibility for tf_pyclif_proto_library intermediate targets. (commit: 81ed5ef2307eea4c9396fd34f33673be072cdcf3)
* Remove unused load statements from BUILD files (commit: d0e01a3c56b280c6602d6c14e97ef60882d317aa)
* Manual tests for model server and including tf.Text in serving build. (commit: 142d0adb5e2975689d80d8fc608c9684e96de078)
* Remove tensorflow/contrib/session_bundle as dependency for Tensorflow Serving. (commit: 1bdd3499f1fe4d99b3c3024080560350d493e29b)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

chaox

# Release 1.15.0

## Major Features and Improvements

## Breaking Changes
* As previously announced[1](https://groups.google.com/a/tensorflow.org/forum/#!msg/announce/qXfsxr2sF-0/jHQ77dr3DAAJ)[2](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md), Contrib ops will not be packaged with Tensorflow, and therefore will not be available in Tensorflow Serving. If serving with Tensorflow Serving >1.15, please ensure your models do not contain any tf.contrib ops. If you are critically dependent on custom ops, please review [this guide](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/custom_op.md) for instructions to statically build ops into the model server.
* After being [deprecated](https://developers.googleblog.com/2017/07/tensorflow-serving-10.html) for multiple years, as a part of tf.contrib deprecation, SessionBundle API will be removed starting from Tensorflow Serving 2.0 - if currently using SessionBundle, please migrate to SavedModel APIs.

## Upcoming Features
* Some Tensorflow Text ops will be added to Model Server starting from TF Serving 2.0.0 (specifically constrained_sequence_op, sentence_breaking_ops, unicode_script_tokenizer, whitespace_tokenizer, wordpiece_tokenizer).

## Bug Fixes and Other Changes
* Add monitoring config (commit: 18db9a46168eadd4d3e28e9b0cdb27bd6a11add9)
* Fix docs (commit: 7fc2253157db1dff340d7b418a6cf5204db2ce09)
* Use master as references (commit: 08cb506672d4c2ef289f79eee545df26d6577b45)
* Fix docs (commit: 9cc986beb742c485a62637fd20e841288774585d)
* Remove hyphen from version numbers to conform with PIP. (commit: 4aa0cfc24098000163fdfe270c4eb205e98790b1)
* Fix ImportError: No module named builtins (commit: e35ffff3999be3f971fa1503c158f33d721228c8)
* Cleanup visibility specs. (commit: 8e3956cac1eec2213538d8d6c367398e2f883e70)
* Remove 'future' from setup.py (commit: 64a80dd955a384de776b6256f3abcaa28cf88e79)
* Install future>=0.17.1 during Dockerfile.devel-gpu build (commit: dc36c21df5117364a3390a8bfe1fd3bf7dc92cb7)
* Replace calls to deprecated googletest APIs SetUpTestCase/TearDownTestCase with SetUpTestSuite/TearDownTestSuite. (commit: 39bbeb70dec8054d8ad81a7aa0423ec7e1a07c2a)
* Add the option to allow assigning labels to unavailable models in open source model server. (commit: e6d91e72f7593be36dda933b3291c7ebbc646fa6)
* Adds polling for config file to model server (#1301) (commit: c3eeed4f245e43f6cf92329d251e2b9d6255d6e5)
* Adds util functions for getting min of two ResourceAllocations. (commit: ba6527c8d56a0752f6c87115071e4d1bf7941810)
* Cleanup usage of the protobuf_archive. See #19032 (commit: dca61db5137c416a454c6ef821ad0fac6d66dc91)
* Replace NumSchedulableCPUs() with MaxParallelism(). (commit: aa9dddb93576c814b97947d6386d400cf6c87679)
* Don't run model_servers:tensorflow_model_server_test under asan (commit: b5c24e3e3849978a551db3aae3854c8794d10124)
* Release notes for 1.14 (commit: dc986268756ef45a3ffca4b8578dfdc69e015d29)
* Fixing Docker link (commit: 3bd851d88cd2febcdec29a52bab1d7d225a3a54c)
* Update release notes for 1.14.0 release. (commit: 00b2980a4d6ca127b63409b3eae791f846d1031a)
* Add release notes for TF serving 1.12.3. (commit: 7226859e9dd0f45bade559ab12892d4e388a7c11)
* Remove unnecessary calls to `Tensor::flat` in the tensorflow regressor interface. (commit: 55d897ef71b1ba142defec67bcce8eba7d8f5236)
* Fix print syntax in sample code (commit: ecef0d2fea2af1d4653a41934649512aa6994fd0)
* Adds guide for serving with custom ops (commit: dae0b4dffb29efc647783d45c28c4db0282b4d51)
* Return more informative error message during warmup. (commit: 1126fcd5d179d7829f48471eca6ddbbce79e219e)
* Enables passing in the SessionMetadata to the TensorFlow Session through the SavedModel ingestion API. (commit: 9cf3ff32daaaa2bb941ba7d7b8f049f807e4288e)
* Modifies server configuration documentation (commit: ee4edd59ad5ea088f1a6616cc6de531f66f25c3d)
* Fixes bazel build errors. (commit: bc07ec9015cba820be7f1087153d216964bd1a0b)
* Add tf.distribute + Keras model save/load test in TF serving. (commit: 093734d002bd9de2a68d34160e23f35db196c080)
* Remove unused fields from MetaGraphDef proto message, stored in (commit: 1f8f2902b6465f239bb58af2b3fb27ba73b5c7c5)
* Fix typo (missing colon) in comment. (commit: 561cabbabe9d44da6b20fcf0eb9170859d3ea9fe)
* Makes ServerCore::Log(...) mockable. (commit: 4b00a803faea0b96c9cbce6fbe23dfaec93bfbd4)
* Uses VerifyFunctionInternal to replace VerifyResourceValidityInternal and VerifyValidityInternal. (commit: b7b3c33422bb5cf0813fdd6604742e7fa3841f84)
* Removed the net_http dependency on absl/base/internal/raw_logging. (commit: 4e12a193ad27fa31cb1e42e9a7fe7b5c08f74c52)
* Override TF defined Abseil version to a more recent version, (commit: 1c0291118f34ec7ba958a9cee594414f6531c0f3)
* Makes VerifyValidity, Normalize and IsNormalized method virtual. (commit: 071634e39f47cde52996c8bfd1ddda8abf4deef9)
* Example of creating tf_serving_warmup_requests (commit: 1623705e4205bc779109f8c4d1eadf6d3f24a549)
* Don't copy SignatureDef. (commit: 28d32a1e487666c8b324c74afb633006ba5cbf17)
* Update resnet_warmup.py example (commit: 00b49bd3f4bcb3b17d1fb61bf302aacccf80c83e)
* Update resnet_warmup.py example (commit: 263025f091dd60d415dd22e9667c0f37f11209ff)
* Instrument BatchingSession::Run with TraceMe (commit: 929ab172ec3553a9d563b13dccfb0926d8bf3724)
* Remove contrib ops from model server from tensorflow 2.0. (commit: e7c987d4b10ac751081c62595fcd18be7481e67a)
* Use C++14 by default. (commit: 41873601c73bcb91e403f9ddd70a168ae117ddb0)
* o Switch to using the half_plus_two model from TF to tensorflow_serving one. (commit: 3ba8a6d8ac31572548bbe7922e4152a6b92e626c)
* Add TfLiteSession class to run inference on TensorFlow Lite Model. (commit: f2407e2011b5fc6d255c0ea54181f9cdd1d691e5)
* Add ability to load+run TF Lite model in ModelServer. (commit: d16ceafa044932e2d9ef84bbe1a6ae5c6356252f)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Abolfazl Shahbazi, chaox, gison93, Minglotus-6, William D. Irons, ynqa

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
