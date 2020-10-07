/* Copyright 2020 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_SERVING_BATCHING_BATCHING_OPTIONS_H_
#define TENSORFLOW_SERVING_BATCHING_BATCHING_OPTIONS_H_

#include <vector>

namespace tensorflow {
namespace serving {

// Batching options.
struct BatchingOptions {
  // If set, restricts the allowed tensor batch sizes.
  //
  // When the batch scheduler forms a batch of size N, the batch size is rounded
  // up to the smallest value M in 'allowed_batch_sizes' s.t. M >= N. The
  // tensors submitted to the "Run()" call are padded with M-N repeats of one of
  // the first N entries (i.e. a guaranteed valid entry). The last M-N entries
  // of the output tensors are ignored.
  //
  // This option is useful when the underlying platform has some per-batch-size
  // overhead, to limit the number of distinct batch sizes that can occur. It
  // may be sensible to use an exponential sequence e.g. [8, 16, 32, ...,
  // max_batch_size], a linear one e.g. [100, 200, 300, ..., max_batch_size], or
  // perhaps a hybrid e.g. [8, 16, 32, 64, 100, 200, 300, ..., max_batch_size].
  //
  // IMPORTANT: The entries must be in increasing order.
  //
  // IMPORTANT: The final entry in 'allowed_batch_sizes' must equal the maximum
  //            batch size parameter supplied to the batch scheduler.
  //
  // If left empty, no rounding/padding is performed.
  std::vector<int> allowed_batch_sizes;

  // If set to true, padding is performed for tensors of the same name
  // but with unequal dimensions (modulo zeroth dimension), so that
  // all tensors of the same name have equal dim sizes.
  // For each tensor its first element is used as padding value.
  //
  // For example:
  // given input tensors of shapes [1, 500, 101], [2, 300, 101], [1, 400, 101]
  // they will be padded to shapes [1, 500, 101], [2, 500, 101], [1, 500, 101].
  // Padding is not performed in zeroth dimension.
  //
  // Supported tensor datatypes:
  // DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16,
  // DT_UINT16, DT_INT32, DT_INT64, DT_COMPLEX64, DT_COMPLEX128,
  // DT_STRING, DT_BOOL, DT_QINT8, DT_QUINT8, DT_QINT16,
  // DT_QUINT16, DT_QINT32, DT_HALF, DT_RESOURCE.
  //
  // Supported ranks: from 1 to 6.
  //
  // This option is useful when using recurrent models(like LSTMs) with serving.
  // These models typically accept variable-length inputs and when
  // training them typical strategy is to save sequence lengths for decoding
  // and pad those variable-length dims to maximum in batch.
  // So, this option is used to achieve similar behavior
  // when serving with batching, it is assumed that sequence lengths
  // have already been saved.
  //
  // If tensors with the same name have different shapes
  // (modulo zeroth dimension) and this option is set to false,
  // then error Status will be returned.
  bool pad_variable_length_inputs = false;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_BATCHING_BATCHING_OPTIONS_H_
