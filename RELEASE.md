# Release 2.18.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Update version for 2.18.0 release. (#2264) (commit: 5815bfdd1d1bbd9d0d3557576c98f13afc4d9016)
* This release is based on TF version 2.18.0.

# Release 2.18.0-rc0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Extend GbmcChannel interface to implement redfish channel for TPUs (commit: 683cb64abb560324c9b1d391cdfe5b56ca1ee25a)
* Add tests to validate monitoring states. (commit: fab5c054d5c4dd18b69e21326367f0c5acae2028)
* Disable xnn_enable_avx256vnnigfni (commit: 19f9ccf9a3ddedc93812da7eac28554ebbc1f8dc)
* Reduce duplicate code using a test class (commit: 51cf3a796d87ed8726bf5525be6481b28de0ef94)
* Define an option to specify different IFRT client. (commit: aca5cfa285061815ab840274264fc6993cd620eb)
* Add release notes for tf-serving 2.17.0 (commit: b72a86e5768017b1699b2c463953d9a5f7db1583)
* avoid SetNumLoadThreads stall the server by forcing reset ThreadPool (commit: 6b9cf7c8777fd79868e73dc07663517993933be0)
* Add max_enqueued_batches option for model servers (commit: 7c99259e82cfdc4f12dbd5715acd6d17fc936b5d)
* Remove gpr_set_log_verbosity from grpc_client.cc (commit: 6e05a385d7f4591d46ae7b1d1a02244a5340a29b)
* Add option to stop retrying on permanent loading errors. (commit: 9ba72fa8a5df6e320caf207bd88673ad4c88e12e)
* Add the batch_padding_policy attribute the tensorflow serving api. (commit: ea02141a00089d77561db46aac0e2ca07bd44b2f)
* Improve handling of large JSON objects. (commit: 6cb013167d13f2ed3930aabb86dbc2c8c53f5adf)
* Silence warnings from external code (commit: 010d61a30f549423f61a3fa29ef0f2f0c8ed7f6c)
* Migration of the histogram header and cc code for TSL. Move tsl/lib/histogram to compiler/tsl/lib/histogram and update users. (commit: ab33df407e103b746aec8e165e31f4bc92ed388c)
* Add hermetic CUDA repository rule calls to TF serving project. (commit: 787c85f1a3f0268a243880418c97f37bed56762b)
* Update users of `status_test_util` to use the new location in `xla/tsl` (commit: 22b2b1e21793c9f7c583a1ee51cf8d73657fb0d2)
* Bump Bazel version from 6.4.0 to 6.5.0. (commit: 82e532fa3a3182560af6f23c38ddcb017c5e384f)
* provide an option to customize the sort order among servable names (commit: 32a85a8b42e6892e380bd4d54cf10b0c5734da4b)
* Remove cc_api_version stage 4: deletion where cc_api_version = 2 (commit: 7e0c1966627d9fa482acf1ef0ea983f4aa90f607)
* Remove cc_api_version stage 4: deletion where cc_api_version = 2 (commit: 48e0f56b8f84310596de1c97037d8d02053a9d14)
* This is a noop comment update for streaming inputs. (commit: cfac240ba956f29b0ae91008d1fe073f94c7ae84)
* Add a resource kind for number of LoRA models. (commit: 6b7ba27fd9dfef8d03fab076ea236e296370a3fe)
* Disable more warnings to make logs cleaner (commit: 4a830cadb604ed3d050b841e280a5b3486f86e4a)
* Add `bool return_single_response` field to `PredictStreamedOptions`. (commit: 648c9ee6489a3cf820aa1fcab82b821209e82af5)
* Use gcc-10 to avoid build issues while building XLA on CI (commit: 8bd1fda7e132a626921e458859db0e519deec451)
* Create separate `kokoro` config (commit: dbc7681fb6b89ed184dc0b41ddcfd59df0bd55b4)
* Remove top-level .bazelrc settings now that scripts use `--config=kokoro` (commit: f920b982ca7341eaf0b6456780d1268a3c8735be)
* Update Dockerfile.devel to build with gcc-10 (commit: f9c0262ecff0425f4647e6d52ab8f346a812e456)
* Move `tsl/lib/monitoring` to `xla/tsl/lib/monitoring` (commit: cb934df6ed2f1dd2b80e71611fe4db3f709dea4c)
* Delete 'enable_lazy_split', since the flag is not used anywhere. The code paths for the above flag being false are retained and true are eliminated. This will ensure that improving batching will be easier. (commit: 873993f9f4c8506194e3a130b0185f71db10bdc6)
* BUILD rule fix. (commit: d89b27235f94b245ae1822b5125f6c67e0b587db)
* Automated Code Change (commit: 4decd0ab78bb3ffb205baeccc64340c2a180ed01)
* Automated Code Change (commit: 0b05e865a05ca6a74344213ce74d0a43f1fbbc40)
* Fix build error (commit: d341c3406f5c4e66525f06fb9232a2ce64d7989b)
* Added capability to use XLA on a GPU. (commit: e5e795f518942a4c61b154a357bc4b16670d3f06)
* Update version for 2.18.0-rc0 release. (#2258) (commit: d6d402263bb9c9dec0151e5aebfa81e5ec015e40)
* Mark Tensorflow compatible with Protobuf v26+. (#2261) (commit: 424dba4101e3d28ac5cf9e65df5747676ef2a1e3)
* Update version for 2.18.0-rc0 release. (#2262) (commit: 67f4ee85350fb48ddda0bd7d1c1ebfd4601ed3e1)
* This release is based on TF version 2.18.0-rc2.

# Release 2.17.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Create separate `kokoro` config (commit: 3af106649212377ef845a9d53fccddb00c10293f)
* Update version for 2.17.1 release. (#2266) (commit: 7b6021dd4cc6c1a815a84f160b77438c84818a66)
* This release is based on TF version 2.17.1.

# Release 2.17.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Add RequestOptions and DeterministicMode options. (commit: a8b200b6363f4761478aa64e345b0822a32065b4)
* Remove usages of bridge fallback. (commit: 98570a6181910017d19990b4be7d9ccf9ae4d174)
* Provide a runtime option to lower bound the number of batch threads. (commit: 50b07e4ceac4f7c0e901f341b5d70b6a61dee736)
* Avoid GetChildren when using Specific servable versions (commit: 6fb94038e62fcc67627729e7f8416e0f146ec3fe)
* Add python clif target for prediction_log.proto. (commit: 39ba6232106cc8bdf99dfb54f5736ad88a17684a)
* Build with --xnn_enable_avx512amx=false (commit: f6c4219a564d8a37bda8ef612b2aca8c3953f304)
* Update comment in tfrt_saved_model_factory.h for wrong param name. (commit: 14ce91115cead955d9ce973866b48373a2ad52db)
* Upgraded libevent to 2.1.12.  Fixed minor bug in EvHTTPServer. (commit: 2cda80ac6c30fe09e41a414669579d2d35460880)
* Introduce RequestRecorder in tfrt_servable so that implementation can record customized costs and metrics. (commit: 749007b9f94da6ab6ef3ed0dcfb3e19829f27298)
* Integrate TFRT+IFRT with tensorflow serving (commit: a8b64dd5e919efce56b52b2dbee39bb884f73296)
* Add core selector support for TFRT+IFRT serving on tensorflow serving (commit: 84a71a42ebdca0a518073ebb02eaac461b582316)
* Remove GPR_ASSERT . (commit: 2dca3af0cb435d00f7af90acbf0eb12db83a3cb8)
* Add timeout support when waiting on servables to load. (commit: 093d841041d3260d4e49ed2401494f0d8b9d1f19)
* Build with --xnn_enable_avx512fp16=false (commit: eeac086b6960923426aaf0eba615a5529bf8f95b)
* Support paging in TfrtSavedModelServable. (commit: 993a53c9a7bd37b70d2db52104bf63daa6b04582)
* Add max_enqueued_batches option for model servers. (commit: d914192fc890811e615be714a0e4769bf9b6dab2)
* Add max_enqueued_batches option for model servers. (commit: 67a2dcb2b9c057847616fb5ce54cb8545955d144)
* Update version for 2.17.0 release. (#2225) (commit: 68eda92e579da6d80d364fde1601504a415be817)
* Include patch files necessary for building at TF 2.17 (commit: 6311b72ea7efdc8c306de3c3bc808388468b7a3d)
* This release is based on TF version 2.17.0.

# Release 2.16.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Update version for 2.16.1 release. (#2210) (commit: 0e6261315e2a8c529842929f5ceeb66b63264e7b)
* This release is based on TF version 2.16.1

# Release 2.15.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.15.1

# Release 2.15.0-rc0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Moves model server TFRT integration code oss (commit: 50ebab4ca601b5243b7aac674628954bef2d734b)
* Add an option to override to the size of GPU system (commit: 445a87ba217a9884bb567d9f9dc33511e08fd519)
* This cl is causing test failures and we are rolling it back. (commit: a39289b422714be4cf8a723d6f7d409f12c0f24c)
* Default signature_method_check to false (commit: 4711a8d4c9dd9d29aa404435e2297aa205914a81)
* Add an option to propagate current Context in periodic functions from AspiredVersionsManager. (commit: e4a8a87cd61bf0a65db948d615e69160e3070201)
* Refactor `Servable::PredictStreamed` so that implementations can support bidirectional streaming if needed (commit: a8c3ea682c57d031f74aae5cb6280e23b3c97b09)
* Create koltin proto library for the tensor flow protos. (commit: cae316414e198b47d86d7d30bc54f3dfd8a05a49)
* Create and use Kotlin proto targets for model.proto and predict.proto (commit: ea9529ea324d6bd76b15152eae7e06cda0b12c49)
* Add release notes for tf-serving 2.13.1 (commit: 45fae91c64f4fefc7e8d3077b2402a964244f0b6)
* Resubmit to move model server TFRT integration code oss (commit: eb5b3a5635e6af87013da86a6ec8c62f1097ec8f)
* Enable BF16 Automatic Mixed Precision (commit: 970c630723061e89e6eb45493c0883b0b97bd7eb)
* Follow expected format (commit: 60a3d7338f714f598071952ecf7e691ba70e3138)
* Remove upper_cost_threshold in TFRT serving (commit: 7f8d9d74bd3e336931526d3fd093aec2a2c66cc2)
* Build tensorflow_model_server with -rdynamic (commit: fc892403efd3d6329a80b11c6a585caec0d72b71)
* Add peak memory resource kind. (commit: 96e0661c2eead9d08e23a3b05c002d05b8cc3c68)
* Fix typo (commit: c0b35c742e51a60c83aee6e073f9488eec1b2326)
* Update warmup documentation (commit: 90148d72e5bfcf6a6977bb578988596c7a98a293)
* Implement Freeze() in pathways/tfrt serving. (commit: 0117fd401cf4063cbc06dc4e34a6a1ce7e3f04ba)
* This CL is a no-op (commit: b75349db95a18c8b906a8502d47a00f296f36c74)
* OSS remote_op_config_rewriter.proto (commit: ba473777a24142d34539505dd7eb85910156cdf0)
* Add release notes for tf-serving 2.14.0-rc0 (commit: 4d5ecfda28cdcb12f4f43a3d73e9386330c26a6f)
* Add flags for gpu multi-streaming support. (commit: 77cabded04612e0415181a790aa005ff0b3c232f)
* Add release notes for tf-serving 2.14.0-rc1 (commit: a3023de03f3064347ee7d14d297efc6fb724f3fa)
* Add 3 new resource kinds constants for GPU. (commit: 6b6dea341396cc2eb88d5a6c9ceac9c788486042)
* Adding flag allowing to turn off automatic TPU system initialization on startup. (commit: f83bc0c6de2b69965dcc0cc4723f21cf68d2a8a4)
* Add release notes for tf-serving 2.14.0 (commit: 60976ef6905e469de56473dd1925db58cdf1c5f9)
* Annotate which model is missing inputs. (commit: c99b18b98827400feb40daec1ce24e33473b7420)
* ebpf-transport-monitoring adding dependency on net_http. (commit: 152ef4ef087533e8608957cd1febff47c2058016)
* Add release notes for tf-serving 2.14.1 (commit: 83d970946861063977e70a0ac110be2651a84caa)
* OSS saved_model_config library, removes saved_model_config_stub/impl, moves GraphRewriter related API from session_bundle_util to graph_rewriter.h. (commit: 7356bbd6e4645b54b70013d9a81a583bdb3fc6d5)
* No-op. (commit: 9d02d895d8562a88974debb25821686218c293a7)
* Upgrading Bazel version from 6.1.0 to 6.4.0 (commit: 34521dc1d8ebce8c18a9acdbab49f0638ecb3398)
* Set xnn_enable_avxvnni=false in .bazelrc (commit: 4aed74931657cce1d12a9217e4ca928482843b4b)
* Add cuda-nvml-dev-11-8 to Dockerfile.gpu (commit: b2def7125180127e4a25212bc8fd7301633104eb)
* Revert problem with incorrect Dart build rules and targets. (commit: b6bccceabd24d449f4c77eb77d34c49b8132459a)
* Add cuda-nvml-dev-11-8 to Dockerfile.devel-gpu (and remove from Dockerfile.gpu) (commit: 028aac5bf94f21c60a74338cddc26a421b556141)
* OSS tfrt_http_api_handler*. (commit: 8ded4cead4fdc0ae6aee0c93991c02c713f1b0eb)
* Added FileAcl to tsl::FileSystem. (commit: d6c0917e5e7fcb76732c7eb172a69075ca2d1b08)
* Remove metadata size check in GetModelMetadata method in order to be consistent with other servable impl. (commit: a6355522da7f539483dc75ab6d995a18e10e9150)
* Replace the global registration with a registration class so that when we move server_init_internal to OSS we won't run into undetermined global registration sequence issue. (commit: 21d8f8816b5871a6bacf4f5ff7da0949d593bf88)
* Move TPU runner init stub to tensorflow serving OSS directory. (commit: 2b9e58cf4ad222adaa741d0cdac3b5a0d220b8fa)
* Add util function to verify if override resource have a subset of device kind of base resource. This is not used by OSS. (commit: 06ff18d38f8fdb955c880ed5200f81189504ae12)
* Add streaming options for predict request. (commit: 8ccd8a5219158294b8183c307809ab76ad6c0f0f)
* Define how tensors will be split for SPLIT streamed requests. (commit: b581572e3d9c1bb708ffa1314c2fd1eb1a143f8b)
* Add a client_id field for custom servables. (commit: eb578521a2a8c349a6602bbe98eb399bdcf20c30)
* Add option to configure the name of the input layer of remote model. (commit: f1e13413e2ad0f0aa8d25fbccaf3cd2abc8dfe0e)
* Added grpc reflection service to the serving binary. (commit: c140e0115a9c6d9cb28279acee0c80d56ed2982b)
* Add the option to enable GRPC health checking to model_server. This is useful for clients that want to use health checking with load balancing channels (if not we get errors on the client side). The current implementation is trivial, once we open our serving port we assume we we always be healthy but users may want to tweak this, specially if they need a mandated version, etc. (commit: a9a8e7bfe982f48a9156dedd48f61f537dea84a0)
* Automated Code Change (commit: f761fc7742d66129e7f8b26bba749d44d4d4b678)
* Update description of model versioning. (commit: d820234866ebb4f8f011f33339b3b9eb7f7333c3)
* Exported `FindMetaGraphDef` function. (commit: 0df0975055046ffc793557b48c68d9e2f16fe406)
* Automated Code Change (commit: 27923d3530cb963eee17d7f3670c9b27499827a0)
* Automated Code Change (commit: 704e2507aa7002d1008ea68f29296c91a6f10267)
* If accepting_requests_ is not set Terminate() returns without doing anything. (commit: c45fe1443b401f58d9440ca5d40c4e407137d042)
* Automated Code Change (commit: fce1804e06a784f9586f7114726c5dc66af1b135)
* Modify PredictStreamed to return a response or an error. (commit: 5b5d30fb9e4d57d3a6f03fc21fcd8e60750de5f9)
* Add support to use a MockServable in MockServerCore. (commit: 5b6e0b6b0806c9175274fb61c3c82a134de036f2)
* Fix OSS cpu build. (commit: 72acbaf554b3ce1c1e91dc86d9dcb859f95e9eda)
* Adds functionality to send TSL metrics over model_service RPC. (commit: 9564ef667dba57887954885328bed627e4c8ea4f)
* Add a method in tensorflow::serving::Servable to indicate whether a servable is critical. (commit: 5c0299e4b7934e5b84c534ed7f3d311fceec38b5)
* Upgrade to CUDA 12.2 and CuDNN 8.9.4 (commit: f82600afc9f22fac59ae10596c21887cbbf4594c)
* Fixes tensorflow_serving continuous build. (commit: a99fb9cbfe527d22f98eedd470a9a00ee0e35920)
* Add headers. (commit: fab72719273e8a6bd4cb1c9f22008269b1d98e83)
* Remove the criticality field in the BatchingSessionTask. (commit: b8663d08b38ad70397f1feb973d98a30c764ba4d)
* Move gpu docker build clang. (commit: 611c5a975681d4fe915274772482450ac6178506)
* Updated Dockerfile.devel-gpu to run setup.sources.sh from repo. (commit: d3102f00871e496a6276beeb546a6b9c46dc27e6)
* Add an interface for all Servables that support paging. (commit: e4716e5aa1b2dfd0b62fdf767de9dffd5d7b3304)
* Update cuda libraries to match TF (commit: 45446cf0f8b82f1bce51627f4005de16f1d597d0)
* Match libraries with Dockerfile.devel-gpu (commit: f6ef270ddf3e1f40064847e2dcf089806874e069)
* Update version for 2.15.0-rc0 release. (#2209) (commit: 73ba2b978678e3768c66d7fa90752da3978fd27b)
* Resolve breakages for 2.15 release. (commit: 318129247a068edbf59dd75e10dbfa2d1eb996dc)

# Release 2.14.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.14.1

# Release 2.14.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.14.0

# Release 2.14.0-rc1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.14.0-rc1

# Release 2.14.0-rc0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Add Intel OpenMP for MKL build since it's required now (commit: 926b08d928d78cda9f371a5d14b2d551e8534631)
* Fixed the broken link in api_rest.md (commit: 146cda04e31ae38e2ae9b9563e639b8c29dd7dd4)
* Add a conditional any_invocable dependency to YDF to allow building with latest Tensorflow (commit: cb3698ef5c81db12cd67088c09eaa4a2dc027db3)
* Internal change. (commit: 62badd2558c792639778d2c517593efbbf26d172)
* Internal Code Change (commit: a1ae333bc7f0436ed0225d3aa1865512a945560f)
* Upgrading Bazel version from 5.3.0 to 6.1.0 (commit: 89a4bda0b9eff7eef1aeca853fdfd1bec5a3fda9)
* Reorder some status checks to break out of while loop immediately on error (commit: a87da7bc53b213ee1c4e14131682a0c31d73b456)
* Enable parallel model warm-up (commit: ba1246d95b9f7fe795bb5ffe449f3cd013552e9d)
* Enable parallel model warm-up (commit: 259f89b5664f4cec2965b819874c24262e3b8678)
* Updates Bazel version from 5.3.0 to 6.1.0 (commit: 9b986296238d1948c47ddfba6d1593dd925f85a6)
* Enable parallel model warm-up (commit: 79f9d2842223faf43db840b5def8e27274a38027)
* Do not register empty model name to warmup registry (commit: 1598cfeda97333b1711f434f016410109ad2dd8a)
* Print out model name for ModelConfig errors, which will make it easy to debug. (commit: 6420cc3381c5c1750635e539f81dde4a6d82a6cf)
* Internal change (commit: 291650abc818f7274c9b72cd410f9aa5fa5c3024)
* Pass criticality and queue option for low priority to support priority queue in shared hatch scheduler (commit: bf4aaf04feb563629cb514497f35e1f84609bb02)
* Add release notes for tf-serving 2.13.0-rc0 (commit: 6a56bd1eefdd46fbfdbd7c906b3fe22929aab3cb)
* Add release notes for tf-serving 2.12.2 (commit: 638fd783f387cc395f5771ec9e05b39b3814145c)
* Add release notes for tf-serving 2.13.0-rc1 (commit: d04bbfeed83801242c2f16c8673592c3ab895aec)
* Internal Code Change (commit: 72b83ed2237cea14bab34d551e01d9dd8fc5edc9)
* Move Servable interface to tensorflow serving. (commit: f196fa8a3c29a309899a8df2058a5692ce999253)
* Add release notes for tf-serving 2.13.0-rc2 (commit: 1ad5e7c89bc32c2fdb8b455d73b361b9accf0e49)
* Add release notes for tf-serving 2.13.0 (commit: 3d8e8c39be54ae88959cb5b9e3b06dc30aadd99d)
* Add RunOptions to Servable interface. (commit: 6a9d0fdafe326605cad1cae60dea0dd165bd2bb4)
* use hermetic python instead of system python for building and testing. (commit: ea05b0f1c274124704dfd3c0a440e9b273e5542a)
* Use OSS-compatible logging lib directly (commit: 3e8dd821e0c469f383e220da137d398168610255)
* Enable the saved model default input value feature for TFRT Predict API. (commit: 45bc2916a5bc5e71f7c46db16567479447f63d7e)
* Adding flag for automatic batch warmup. (commit: 736e1298a66b0976bf3a554492c74cb23db8904d)
* Internal Code Change (commit: 008c02c45e8a09b9a82d80b0962b5918e759f10e)
* Support streaming in request logger. (commit: 2cdb6b940378676e031d7b0c18f32a1a063c0f77)
* Use Servable for TFRT serving (commit: 3863ae35a69a1da616d94edcdc4f736cd046fddd)
* This is a noop (commit: 444257dc8cf3a9c69b3c8036821c2c9c1d73a670)
* Update version for 2.14.0-rc0 release. (commit: d07fc99ef40ebc044c27a04a20ff976ff9e2847c)
* Update Dockerfile.devel* with py3.9 installed. (#2178) (commit: 89e76e785ebe458264e6a40b76b8258e32377103)
* Update WORKSPACE to use python 3.9 (commit: 24b94ecab15e70afe2b546f5f7a093da6200569f)

# Release 2.13.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.13.1

# Release 2.13.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.13.0

# Release 2.13.0-rc2

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.13.0-rc2

# Release 2.13.0-rc1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.13.0-rc1

# Release 2.13.0-rc0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* Remove unused TF-DF define from bazelrc. (commit: dc0766022a49e8eca1ad9d7c7ce212a8ee3ea6d7)
* Remove usages of `tsl::Status::error_message`. (commit: 2b4ef1ae5ef3d24bdfa994dc08e840d635c66125)
* Add release notes for tf-serving 2.12.1 (commit: 19c3e66306722c44ddc7fc6dd2b4c59373640c95)
* Fix tests broken by the new defaults info in SignatureDef. (commit: 07b0882570a040aeb4a74a01ec5c81ecd6383afc)
* Fix remaining tests broken by the new defaults info in SignatureDef. (commit: 65b13b76083adfa7b694b5378e3f91a5088e62b8)
* Deprecate oss_or_google.bzl, move macros to serving.bzl (commit: a85461300e0e6a4b589e809bb7390ea4f17796bd)
* Revert "Fix tests broken by the new defaults info in SignatureDef." (commit: 663b375d8899e5937e67655f56ee6c888f4a084a)
* Update version for 2.13.0-rc0 release. (#2151) (commit: f449ae3dc99bfd19a3e9dbfa213c8553222382e2)

# Release 2.12.2

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.12.1

# Release 2.12.1

* This is a re-release of 2.12.0 (that was marked as bad). Please use this instead.

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* Users of remote_predict_py should stay on v2.11 and avoid v2.12.

## Bug Fixes and Other Changes

* Update TF Serving pip package protobuf requirements to match [TF's](https://github.com/tensorflow/tensorflow/blob/0db597d0d758aba578783b5bf46c889700a45085/tensorflow/tools/pip_package/setup.py#L107). (commit: 24028778d11bf67992d481ff573de171c396119b)
* Update version for 2.12.1 release. (#2139) (commit: bd203faa888dd5ce90f21e3ee9af92dbc90b8a25)

# Release 2.12.0

* **NOTE:** 2.12.0 has been identified as bad release. Please use 2.12.1 or later instead.

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* Users of remote_predict_py should stay on v2.11 and avoid v2.12.

## Bug Fixes and Other Changes

* Update TF Serving to Bazel 5.3.0, to match with TF. (commit: cda26f6065753167ac83e3b1aad7485d3d1d6db0)
* Update TF Text to v2.11.0. (commit: 1624fb20014921eac178318294e3b0c40d583d4e)
* Add pyclif_proto_library for get_model_metadata and session_service (commit: bedf391e8617cd3b1cf01ac5efd9e2fe8543a6ca)
* Raise the vlog level about aspired versions (commit: 73746fb3adeb29f8a1f20e154e8480397afd593d)
* Add PredictStreamed to PredictionLog. It represents a logged stream of PredictRequests and PredictResponses. (commit: 557f68a88d6be9a0f53beeca02a366359d787d4d)
* Update rules_pkg to 0.7.1. (commit: 2b5ad9a0d0424f285cc5e1a11eaeb5a8a0c89ad2)
* Track additional metadata for request logs. (commit: e0a3b5f990b9801d21c739dd27b8430c49353d8b)
* Replace usage of the tsl::Status constructor with a tsl::{error, errors}::Code. (commit: cfb9fb221e3375fe1dee144a70a9a2f4e28b01da)
* Replace usage of the tsl::Status constructor with a tsl::{error, errors}::Code. (commit: 19345a666becbe2df1d2d6096cae88b9013848ad)
* Update Dockerfile.devel* with py3.8 installed. (commit: 68d92ff3fdca0641f465cc3ba3858a619c8b82a6)
* Update TF Text to v2.12.0. (commit: dbe9339b436b2fa20705c8a444230848e771d65b)
* Stop depend on 'tensorflow-gpu' for tensorflow-serving GPU build on master branch. (commit: 85ff9c06021b47d487807a48a645d1c6ee9f654b)
* Upgrade cuda from 11.2 to 11.8. (commit: 20f51c91c0c19f1836508bec8ab7764d208f8f7f)
* Upgrade to decision-forests-1.3.0 and yggdrasil-decision-forests-1.4.0 (commit: 45d157458cbff136666700b6dee5fc7ccfe0dc70)
* Add tool to send predict (grpc) requests to ModelServer. (commit: 25c51251ed09ec80bf1d8380296649f9e1770e7b)
* Ignore remote_predict from :all for 2.12 release due to upstream TF breakage. (commit: 5830714e4f831e90b54f40e3f0467cac74caa009)

# Release 2.11.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.11.1

# Release 2.11.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.11.0

# Release 2.11.0-rc2

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.11.0-rc2

# Release 2.11.0-rc1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.11.0-rc1

# Release 2.11.0-rc0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* No-op for public code. (commit: 20655b5904fdb4a810bfd8aef22db8becd4e80f3)
* Removed net_http/client/public/* now that all the users have been migrated to test_client/ (commit: e64a73afc23ea8f9538660cb8bc20f48d5682848)
* Fix the bug when accessing concat.tensor_data() outside its lifecycle (commit: cc3720091867c06615b8b3f50c1c2468a90e6b1c)
* BUILD cleanup (commit: 9e81e7b1bc8e1bd28c3324cf8feee717420f96f4)
* BUILD cleanup (commit: 8235ed86f9f07123e30ea8830a1454334bc86a2c)
* Update TF Text to v2.8.2. (commit: 31204d664d4ff985a526dccfe1f13ffde113deb6)
* Replace tensorflow/core/lib/core/blocking_counter in favor of the one from tensorflow/platform. (commit: 4ed7c5c30d0be43254a1177e8488675dc60d0384)
* Update TF Text to v2.9.0. (commit: e88506ab6709f2dc3a3bb1601ac4b84c23025611)
* Cleaning up BUILD files to remove "loose" headers. (commit: c2cd4c9629fbc868bd5e2cb186c7b081f22d709b)
* Remove the unnecessary dependency (commit: d976a37b4e9481de5ca371ae734645d925143088)
* Use `value` instead of `ValueOrDie`. (commit: b48e6e7cc02d8060cc3da5d363a90baf9324a2a0)
* Update TF Text to v2.9.0. (commit: a73f2925e367106bab0bfeca187fbaa1b3f36676)
* Replace `tensorflow::Status::OK()` with` tensorflow::OkStatus()`. (commit: 49ac8acb50291c21a0a72cfd9135aa2030e3ae88)
* Replace `tensorflow::Status::OK()` with` tensorflow::OkStatus()`. (commit: e8be1a742c3913cbbb6158dc1202c17130583219)
* Replace `tensorflow::Status::OK()` with` tensorflow::OkStatus()`. (commit: 88bd9d1638f6017b4af526a9b468641966d8972d)
* Add support for TensorFlow Decision Forests models. (commit: 4592081169068a0f059be71ac1b484d568f6e5d2)
* Remove special handling of tflite model when creating batching (commit: 2d39f8c8aa90ccfdc78faa51e4c8295832796a68)
* Improve batching test in following ways: (commit: 68d4d3962dc40fa6c1980df6d8df51338e44301e)
* Split proto helper function to its own util library. (commit: 01386268be6948429ea796fcd348beea4c7174d6)
* Created net_http/public folder for shared files and updated files in net_http and model_servers/http_server.cc to match. Also removed old directories from net_http/client to make way for new client API and implementation. (commit: 8086d33b6b74899dc8062ddbdba3fcc95e9af7e5)
* Updating a reference to Env post-refactor. (commit: 728474f92533be737958edc095abc0648a494452)
* Replace `tensorflow::Status::OK()` with` tensorflow::OkStatus()`. (commit: 9861f7fd742dc3ec0a5e2cf0314d68b799a11e73)
* Experimental support for per-model batching params. (commit: 94f2d6944f310069ebd527a52649fe3c001c38e3)
* tensorflow-serving-api python package requires Python >= 3.7 (support for prior versions has been dropped). (commit: 30ea18d66416e8cfd3873fba3c2482ec68184ec6)
* Replace `tensorflow::Status::OK()` with `tensorflow::{Status, OkStatus}()`. (commit: 6393b29242ab6648c723a51f64eb2e67ebf497db)
* Open up http_server_clients package group for easier 3P (commit: 66b199ba6e8fc81cb278988aa541e03987a1d27d)

# Release 2.10.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.10.1.

# Release 2.10.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.10.0.

# Release 2.9.3

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.9.3

# Release 2.9.2

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.9.2

# Release 2.7.4

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.7.4

# Release 2.8.4

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.8.4.

# Release 2.8.3

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes.

## Bug Fixes and Other Changes

* This release is based on TF version 2.8.3.

# Release 2.10.0-rc3

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.10.0-rc3

# Release 2.10.0-rc2

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.10.0-rc2

# Release 2.10.0-rc1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.10.0-rc1

# Release 2.10.0-rc0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* tfs:aarch64: add aarch64 mkl bazel config to enable onednn+acl backend (commit: 1285e41acc707ba0d18e8eaf8a42c6d5110e8af8)
* Match packages in devel and non-devel GPU with TF dockerfile. (commit: a8ffec79e0794650a4c0856c4122032e985296cc)
* Validate batching params when creating the wrapped (batching) session, (commit: 48ff72dcb6582e989452ba870c88a2bb710ea0c4)
* Merge and clean up implementations of `GetModelDiskSize` and `GetAllDescendants` in util.cc. (commit: 6da9c43c5f71abe361841fb3fd5eaad57fc847b1)
* Parallelize iteration over child files in `GetModelDiskSize` (commit: d09d2efe6e9b88ef0266e5982a3e732da14dc93b)
* Fix gpu docker build failure due to bad substitution (commit: 1d7cd5b2ba43c3d98f0c8bef6806c203b2c14592)
* Call run_returning_status instead of run (commit: 8f9085ac678755afea6bf0067fe40a32e37ce2fa)
* Fixing broken link for [ResNet in TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet). (commit: b15210ca076b11eaa2dfd0ac2fb125be780c5d40)
* Update the TensorFlow BatchingSession metric monitoring class to be compatible with Google's internal monitoring tools. (commit: 05b944ad9367027a1082756a3069619f44955de1)
* Increase timeout for model tests. (commit: 677ba5a07813c4fb5a2ffb4567a7ec4a137eebe6)
* Use pb.h for topology.proto. (commit: 21fda280bc72bdbc4386c7b0d2ad4b97264921ad)

# Release 2.7.3

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Update TF Text to v2.7.3 (commit: ee7892be7801a0e4ae9a6dd8b5f7bab06ae9c87c)
* This release is based on TF version 2.7.3

# Release 2.9.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.9.0

# Release 2.6.5

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Rollback incompatible C++17 changes. (commit: ba0fa72b61bc2c42388b815253ba72e6830f03cf)
* Roll forward with std::optional -> absl::optional. (commit: 009dac683bf84165b84702d177bb9a021ebef728)
* Replace STL algorithm call with a container method (performance-inefficient-algorithm). (commit: f5bc09998e0043ce72d34b14104379163048406c)
* Remove unused "using" decl. (commit: ffcc4a16c76c4fa1189edd9682fc486442a33e52)
* Move status_proto to public visible apis/ (it being used by public API protos) (commit: 7f894c79fce5e58758f3cb49e858a16e3602ae80)
* Move core/logging.proto -> apis/logging.proto (commit: 37c64b8820a923aafc1b5c8bf264fd5cce5224f3)
* Update TF Text to v2.5.0. (commit: 48e5a9e23a1e0b2951b77c3e8f9832193d9b1851)
* Adding python targets for config protos (commit: 757c3a6b6c8a03731dc73ff758f69a61aeddcf67)
* Remove experimental tags from uses of gRPC C++ callback API. (commit: b355023b034ca6ef72b507920d9de2a02e0f4a2a)
* Add new --use_alts_credentials flag, to enable building secure credentials using Google ALTS. (commit: ceba636bb7c5c98bde35d1818fd033b36300fffe)
* Enable HTTP PATCH support in evhttp_server (commit: 6cbc4a9eb419c8078c3a4e791381cda70dd8fc78)

# Release 2.9.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Do not report successful loads that are cancelled, unload immediately instead. (commit: 923269955f5582cc26d0454992afa5c888a9377f)
* Add a lock to avoid race condition on memoized_resource_estimate_. (commit: 83959e364e7ff1234bf47a5d8139677d1bdb18c1)
* Update Resnet model used in K8S tutorial (commit: 6e76f6a9460bf2d37630f025fcfd3e06c4230fee)
* Prepare for development of new http client API. (commit: 78e94e99650deae956fe20dffa9932a72ec7d765)
* Integrate TPU initialization changes into TF Serving. (commit: 6549ef2136940cd98cfbb9ee0e29221d86101d16)
* Allow max_execution_batch_size to be actually used by allowing (commit: 48a699a2fd32404c4b19f55077a1fb29112a0afe)
* Delete batch_schedulers_ before thread_pool_name_ (commit: 2837812341e7c98be4717e5901286692a5dcc02a)
* Add missing NVIDIA repository key. (commit: c0998e13451b9b83c9bdf157dd3648b2272dac59)
* Bump minimum bazel version 5.1.1, to match with TF and root.workspace (commit: 8a02874cee6957e5817960613627a549bd80a6e9)
* Update to use C++17 (commit: 7166e2efc6b7e63c908515c6a53d0e4fe8fa0aae)
* Update tensorflow_model_server_test to depend on the pip installed tensorflow. (commit: 04b8676730a882cab61b7772097f2c23c0447ef9)
* This release is based on TF version 2.9.0

# Release 2.8.2

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Replace int64 with int64_t and uint64 with uint64_t. (commit: 21360c763767823b82768ce42c5c90c0c9012601)
* update to latest benchmark API changes (commit: 860e1013385289ad3f9eb4d854b55c23e7cb8087)
* This release is based on TF version 2.8.2

# Release 2.8.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Force-allow labels for the models which are not yet loaded. The feature is meant to be used for non-prod environments only as it may break the serving until relevant models are not loaded yet. (commit: 988bbce80038ac0b7141dcb4413124ba486344cf)
* Update CreateRPC API interface. (commit: 2e7ca90c18f310c542ed0dcde92d676db6454285)
* Add `--tensorflow_session_config_file` flag to tf serving model server to support custom SessionConfig (commit: 342a8088524c33f68e3eb4d66800f01a777ceb38)
* Add `--experimental_cc_shared_library` by default to all builds. (commit: 49b138fdd4c0fb7170736193063c6f03dfb4dba4)
* Add --num_request_iterations_for_warmup flag (fixes #1949) (commit: 2a55aec18cc1923ece84c7fcf701185306ef99b1)

# Release 2.5.4

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.5.3

# Release 2.6.3

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.6.3

# Release 2.6.0

## Major Features and Improvements

* Update TF Text to v2.5.0. (commit: 48e5a9e23a1e0b2951b77c3e8f9832193d9b1851)
* Add support for Google ALTS. (commit: ceba636bb7c5c98bde35d1818fd033b36300fffe)
* Enable HTTP PATCH support in HTTP/REST server (commit: 6cbc4a9eb419c8078c3a4e791381cda70dd8fc78)

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Enable tensor output filters with batching. (commit: 448dbe14624538ab76fd6aeb2a456344e7f41c78)
* Update tf.io import in warmup example doc. (commit: 6579d2d056530565cd6606a39c82b2f6c1d3799e)
* Resize tensors if the dimensions of the tflite and tensorflow inputs mismatch, even if the number of elements are the same (commit: 8293f44bd5c5ecc68636cd0d036234f891d29366)
* Add basic batch scheduler for tflite models to improve batch parallelism. (commit: 0ffd6080437ca8175b067be7cc00f5b3df9ea92a)
* Reserve Unavailable error to TF communication ops only. (commit: db9aca187affd0453627a1729916acfea98ae800)
* Add the flag thread_pool_factory_config_file to model server and fix a typo. (commit: efc445f416f8cb20606ca0d2aaf44c13fae7ea4c)

## Thanks to our Contributors

This release contains contributions from many people at Google.

# Release 2.5.2

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.5.1

# Release 2.4.3

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.4.3

# Release 2.3.4

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* TensorFlow Serving using TensorFlow 2.3.4

# Release 2.4.1

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* This release is based on TF version 2.4.1

# Release 2.4.0

## Major Features and Improvements

* Update TF Text to v2.3.0.
* Upgrade to CUDA Version 11.0.
* Update CUDNN_VERSION to 8.0.4.30.
* Adds user guide for Remote Predict Op.
* Add support for serving regress/classify (native keras) TF2 models.

## Breaking Changes

## Bug Fixes and Other Changes

* Adding /usr/local/cuda/extras/CUPTI/lib64 to LD_LIBRARY_PATH in order to unblock profiling (commit: 1270b8ce192225edcaafb00a50822216dd0b1de0)
* Improve error message when version directory is not found (commit: d687d3e8827c82f4f1b68337c67b2cbe6e4126e7)
* Migrate the remaining references of tf.app to compat.v1. (commit: 06fbf878a98c8bd4202e33bc1c097a6ce184d06e)
* Cleanup TraceMe idioms (commit: f22f802c73bfdd548f85dacffc24022b0d79dfc7)
* Adds LICENSE file to tensorflow-serving-api python package. (commit: 41188d482beb693d4e79e6934d25f1edd44321ac)
* Enable a way to 'forget' unloaded models in the ServableStateMonitor. (commit: 53c5a65e8158dc1a2a85a2394482cc6acc1736bc)
* Added abstract layer for remote predict op over different RPC protocols with template. (commit: c54ca7ec95928b6eec39f350140835ebbe3caeb0)
* Add an example which call the Remote Predict Op directly. (commit: d5b980f487996aa1f890a559eae968735dfebf5d)
* For batching session in TF serving model server, introduce options to enable large batch splitting. (commit: f84187e8d3e19a298656a661a888c0563c21910e)
* Add multi-inference support for TF2 models that use (commit: abb8d3b516a310ec4269cd6bf892644d5150485a)
* Use absl::optional instead of tensorflow::serving::optional. (commit: c809305a50412a2b47f2287c76ea0be1070aabd6)
* Use absl::optional instead of tensorflow::serving::optional. (commit: cf1cf93eac1896c3c482d17b440489edea110670)
* Remove tensorflow::serving::MakeCleanup and use tensorflow::gtl::MakeCleanup. (commit: 6ccb003eb45f4961128e5cc2edf5d8b61ef51111)
* Use absl::optional and remove tensorflow::serving::optional. (commit: e8e5222abbb39e84d1d4e5e9813626b2cc51daac)
* Deprecate tensorflow::CreateProfilerService() and update serving client. (commit: 98a55030e10a61ee0c3f6b8fc57e2cf63fc59719)
* Change the SASS & PTX we ship with TF (commit: 086929269b5f2c0f5d71c30accb79d74694c9ece)
* Adding custom op support. (commit: 892ea42864676b67cbccdfa0794a15d30e65a1b6)
* Upgrade to PY3 for tests. (commit: 02624a83f70060095df7c132fa46a7a09f9bff6a)
* Makes clear how to make a default config file for serving multiple models. (commit: 084eaeb15fdc87d83b8c19f558dc1f56bd3a024e)
* Use TraceMeEncode in BatchingSession's TraceMe. (commit: 78ff058501274aa37b6bbc18aec225604d4cda47)
* Export metrics for runtime latency for predict/classify/regress. (commit: c317582981cfc1550b27d9d73f71c6ca38e5c8c5)
* Refactor net_http/client to expose request/response functionality as a public API (not yet finalized) for usage testing ServerRequestInterface and HttpServerInterface instances. (commit: 0b951c807375f1f305280a96124d8b6d6e045bd2)
* In model warm-up path, re-write error code out-of-range (intended when reading EOF in a file) to ok. (commit: d9bde73569385b4ef3ef8e36d2c832a8ae9a92ad)
* fix Client Rest API endpoint (commit: b847bac5f2e1dc6a98f431b1fdf42ceebceceeb6)
* Support multiple SignatureDefs by key in TFLite models (commit: 2e14cd9dc2647127d7cb8c44ceab5dfcf6ac28c4)
* Add dedicated aliases field to ModelServerConfig. (commit: 718152dc386f9fa7b21ed36d9d85518e987d7bf5)
* Remove deprecated flag fail_if_no_model_versions_found from tensorflow serving binary (commit: 4b624628977a12b1757b9ddcd3312b3768de8231)
* Fix TraceMe instrumentation for the padding size. (commit: 0cb94cd79aacb965b3923d4a51b4091cf84d5e22)
* Add vlog to dump updated model label map (for debugging) each time the map is updated. (commit: ac10e74078123189dc1c8a3cd29d530b7c972782)
* Add python wrapper for remote predict op and clean the build and include files. (commit: d0daa1064ecdd56ecb5c0a8aca37c3e198cb313d)
* Add `portpicker` module required to run modelserver e2e tests. (commit: 82f8cc039d091916b8186dfa1ff4b6c006e7277c)
* changing "infintiy" to "really high value" (commit: c96474cfcca46b1216e52634efb68986cf8aa9b8)
* Minimal commandline client to trigger profiler on the modelserver. (commit: c0a5619a01e3af69459aa6396d614945370bbd02)
* Add signature name to RPOp. (commit: 84dfc8b66ff6c1a693766613034ddc3ff044a330)
* When RPC error occurs, the output tensors should still get allocated. (commit: 9113de22353350443bdd42c5d594ec653e57c0da)
* Fix BM_MobileNet benchmark (commit: af665627b8152d4c62d207a97c6e712cb2e9a120)
* Add JSPB BUILD targets for inference and example proto files. (commit: f1009eb0e6bdae2e35dbfb9f4ad7270e74705e2e)
* Fall back to legacy TFLite tensor naming when parsing signature defs in TFLiteSession. (commit: 3884187cb9253bb9baa240b2009cfc6d4847b9f9)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Adarshreddy Adelli, Lescurel


# Release 2.3.0

## Bug Fixes and Other Changes

* Add a ThreadPoolFactory abstraction for returning inter- and intra- thread pools, and update PredictRequest handling logic to use the new abstraction. (commit: 8e3a00cd8ef00523227cbe1f694ab56454a880c3)
* Update Dockerfile.devel* with py3.6 installed. (commit: b3f46d44d07480266b28776caa13211339777bc5)
* Add more metrics for batching. (commit: f0bd9cf8b85710b638938361d356dbf15fda2e86)
* Rename method to clarify intent. (commit: 9feac12f2223124c7ecc85a687e1ee2b24e3f7ad)
* Plug ThreadPoolFactory into Classify request handling logic. (commit: 975f474a4ea9ef134439e266ec4a471741253ecf)
* Plug ThreadPoolFactory into Regress request handling logic. (commit: ff9ebf2db8bf7cbc7bb199bbb207409eae25d5cc)
* Plug ThreadPoolFactory into MultiInference request handling logic. (commit: 9a2db1da9b7e992d29ad4ccfcb125734d0cd760e)
* Add a tflite benchmark for Mobilenet v1 quant (commit: e26682237cf756eca2dc12c83e8d5d24f00c1261)
* Allow batch size of zero in row format JSON (commit: fee9d12070a76c1cf56bc8ae40f306a09dfd07b1)
* Add tests for zero-sized batch (commit: b064c1d3df03b0401c5ca61de0d5ab36cd5645a5)
* Support for MLMD(https://www.tensorflow.org/tfx/guide/mlmd) broadcast in TensorFlow serving. (commit: 4f8d3b789964d173f2d0bd87a42abfbd6a2b1e71)
* Fix docker based builds (fixes #1596) (commit: ca2e0032d1ead843398d7744e8c51ead28daf63c)
* Fix order dependency in batching_session_test. (commit: 58540f746c65516dc3fcda7751c6983050307409)
* Split BasicTest in machine_learning_metadata_test into multiple test methods without order dependency. (commit: 745c735e315941925e324cbebe78a1f09d5a7443)
* Revert pinning the version for "com_google_absl". (commit: ff9e950fa692c6f9387239bb9fa877975e8cf1c1)
* Minimize the diffs between mkl and non-mkl Dockerfiles (commit: e7830148e53acfec7d3af7dd512a7e825f75da2a)
* Pin "com_google_absl" at the same version(with same patch) with Tensorflow. (commit: f46b88af8af94be3c6497cc6c50a4e5c0625b2d5)
* Update TF Text to v2.2.0. (commit: f8ea95d906421ff9517b0027662546741c486edf)
* fix broken web link (commit: 0cb123f18df4032d8f22c1b2e19b4f41bd6c3da3)
* Test zero-sized batch with tensors of different shapes (commit: 1f7aebd906a70ba0fa04105ceee6227960b764f7)
* Test inconsistent batch size between zero and non-zero (commit: 91afd42dab8ce50f86bbf65065dce0c28163314b)
* Fix broken GetModelMetadata request processing (#1612) (commit: c1ce0752f1076bd6f92e1af5f73e3a3c552f4691)
* Adds support for SignatureDefs stored in metadata buffers to tflite sessions (commit: 4867fedbff8a33f499972268abe96618abcb81aa)
* Update ICU library to include knowledge of built-in data files. (commit: c32ebd5e9f09828c80413ca989b99e8544502c1a)
* Add support for version labels to REST API (Fixes #1555). (commit: 3df036223b66738de1b873e9b163230fb7661cb4)
* Update TF Text regression model to catch errors thrown from within ops. (commit: 425d596b9b0aef2bf3ea675c985f01e55f880a4e)
* Upgrade to CUDA Version 10.1. (commit: fd5a2a2508daf21ad174b4ec7b62501486137c01)
* Migrates profiler_client trace to the new api in tensorflow_model_server_test. (commit: 8d7d1d6bbc50756e73aed4b9eb5a2c8ff25cdc79)
* Update the testing model for TRT to fix the test. (commit: 28f812d8ce8f256e2d9256d6a98cd8f75f747842)
* Add release notes for TF Serving 2.2.0 (commit: 54475e6508889c13992aced1da12a372d997e4e3)
* Update bazel version requirement and version used in the docker images to match with TF (3.0.0). (commit: 56854d3fa27cce8c1f7816214f59e6e82c4bf5fc)
* Fixes instructions on sample commands to serve a model with docker. (commit: a5cd1caafacd7480f5d8d2dd164adce3410b024f)
* Change use_tflite_model to prefer_tflite_model to allow multi-tenancy of Tensorflow models with Tensorflow Lite models. (commit: 8589d8177bd300625b4c7596240150f8a8002d19)
* Introducing Arena usage to TensorFlow Serving's HTTP handlers. (commit: a33978ca4c29387845e9b51d5653b997d4b3f814)
* Fix tensorflow::errors:* calls, which use StrCat instead of StrFormat (commit: 2c0bcec68c040306e009b5a10d4bc80bc58fe0c5)
* Instrumentation for BatchingSession: (commit: 3ca9e89d1b6147706981467a84c6421c44d3794a)
* adjust error message for incorrect keys of instances object (commit: 83863b8fec26a8ea2d3957366173f9a52658b469)
* Update rules_pkg to latest (0.2.5) release. (commit: 932358ec7511e54ad9c93ea606cc677da2d1fcb2)
* In batching session, implement the support for 'enable_large_batch_splitting'. (commit: d7c6a65b816849cf2b84015a5b2972be7950dc89)
* Update version for 2.3.0-rc0 release. (commit: 3af330317628a713a6e318097c7cd6fa8571165d)
* Set cuda compute capabilities for `cuda` build config. (commit: 731a34f0b3f43a6f7a8da85655d3a4a5c72d066a)
* Update version for 2.3.0 release. (commit: 8b4c7095b9931442a77288624fdd1a207671eb4c)

## Thanks to our Contributors

This release contains contributions from many people at Google.


# Release 2.2.0

## Major Features and Improvements

* Upgrade to CUDA Version 10.1. (commit: 3ab70a7811f63b994da076e2688ccc66feccee96)
* Update TF Text to v2.2.0. (commit: fd9842816eddb4782579eadd119156190d6d2fec)

## Breaking Changes

## Bug Fixes and Other Changes

* This release is based on TensorFlow version 2.2.0
* Add a SourceAdapter that adds a prefix to StoragePath. (commit: f337623da81521eefd8cdc2da1c4a450ecf1d028)
* Switch users of `tensorflow::Env::Now*()` to `EnvTime::Now*()`. (commit: 8a0895eb8127941b2e9dada20718dd28f3dbaee1)
* Remove SessionBundle support from Predictor. (commit: 2090d67f4e5e8ee5aa7faf8437bea096a438450a)
* Replace the error_codes.proto references in tf serving. (commit: ab475bf6c5e5e4b3b42ffa2aecf18b39fd481ad3)
* Adds [performance guide and documentation](tensorflow_serving/g3doc/tensorboard.md) for TensorBoard integration (commit: f1e4eb2259da90bb9c5fe028ba824ac18a436f67)
* Remove SessionBundleSourceAdapter as we load Session bundles via (commit: d50aa2b0b986b11368ddcf6b6eb20b9381af474c)
* Use SavedModelBundleSourceAdapterConfig instead of (commit: 8ed3ceea985529a350290cf782cb34c3c66827d4)
* Update minimum bazel version to 1.2.1. (commit: 1a36026198df5f7dec1e626ef9b112fecdd2916b)
* Drop support for beta gRPC APIs. (commit: 13d01fc64330ff883bd1553122d9fd114a5a7368)
* API spec for httpserver response-streaming (with flow-control). (commit: fd597f074ce127056515bc52ee3a3d4ff4b727bb)
* Change Python version to PY3. (commit: 7516746a311f96b57a60598feba40cbdd3989e73)
* Update Python tests in PY3. (commit: 0cf65d2105c191c00fba8918ba75fc955bbeace3)
* Upgrade bazel version for Dockerfiles. (commit: e507aa193b9f3520d40e3da5e4d2263280ff35e4)
* Change dockerfile for PY3. (commit: 7cbd06e8b7720b82b1d2dfae54c3a828d3a52eb4)
* Reduce contention in FastReadDynamicPtr by sharding the ReadPtrs, by default one per CPU. (commit: d3b374bc70348f2e5e22b7e9ebb191ee9d5b3268)
* Ensure that all outstanding ReadPtrs are destroyed before allowing a (commit: e41ee40826652b6aa5a3f071107074923d6ff6c7)
* Allow splitting fields from batched session metadata into individual sessions (commit: caf2a92ba07ca4d10515f0b018c920e9b325c6c8)
* Allow passing ThreadPoolOptions in various Session implementations. (commit: 2b6212cf0aa88b719ee00267f83c89d4f7599ef1)
* Update bazel version used in the docker images. (commit: 162f72949c6ecbe9e610182c923dec0aa5924cf2)
* Format error strings correctly in JSON response (Fixes #1600). (commit: 1ff4d31cd9a0a736162813c149139cce0ccaaa2c)
* Fix broken GetModelMetadata request processing (#1612) (commit: 55c40374b548b89e8de6d899ef2b0b355c0fa9e5)
* Support Python 3.7 in tensorflow-serving-api package (Fixes #1640) (commit: f775bb25e80a6c7b3c66842eb9085d44d9752ec2)
* Update ICU library to include knowledge of built-in data files. (commit: 774f2489384cf985c534298d1303474c268efe5c)
* Adds storage.googleapis.com as the primary download location for the ICU, and resets the sha256 to match this archive. (commit: 028d05095c4e302c06096e5ea32917718828ea47)

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
