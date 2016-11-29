/* Copyright 2016 IBM Corporation. All Rights Reserved.

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

#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/insert_splits.hpp"
// note: this header resides in third_party/caffe
#include "openblas_prelude.h"

// avoid fp-16 redefinitions when using
// a recent version of cuda
#if CUDA_VERSION >= 7050
#define EIGEN_HAS_CUDA_FP16
#endif

#include "tensorflow_serving/servables/caffe/caffe_serving_session.h"
#include "tensorflow_serving/servables/caffe/caffe_py_util.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_set>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace serving {
namespace {

// Constructs a flat tensor with 'vals'.
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals) {
  Tensor ret(DataTypeToEnum<T>::value, {static_cast<int64>(vals.size())});
  std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
  return ret;
}

// Constructs a tensor of "shape" with values "vals".
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals, const TensorShape& shape) {
  Tensor ret;
  CHECK(ret.CopyFrom(AsTensor(vals), shape));
  return ret;
}

// A guesstimate of the batch size; assume the outermost
// dimension of the input blob(s) indicates the batch size,
// unless the input is 1-dimensional, in which case assume
// batch size of 1. (I couldn't find much concrete
// documentation on this.)
unsigned int BatchSizeOf(const caffe::Net<float>& net) {
  unsigned int x = 1;
  for (int idx : net.input_blob_indices()) {
    const std::vector<int>& shape = net.blobs().at(idx)->shape();
    if (shape.size() > 1 && shape[0] > 0) {
      x = std::max(x, (unsigned int)shape[0]);
    }
  }
  return x;
}

// Parse GPU ids or use all available devices
void GetGPUs(std::vector<int>* gpus) {
#ifndef CPU_ONLY
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  for (int i = 0; i < count; ++i) {
    if (caffe::Caffe::CheckDevice(i)) gpus->push_back(i);
  }
#endif
}

bool TryAssignGPU(const int force_device_id, int& device_id) {
  using namespace caffe;

  std::vector<int> gpus;
  GetGPUs(&gpus);

  if (gpus.size() != 0) {
    if (force_device_id >= 0) {
      if (std::find(gpus.begin(), gpus.end(), force_device_id) != gpus.end()) {
        device_id = force_device_id;
        Caffe::SetDevice(force_device_id);
        Caffe::set_mode(Caffe::GPU);
        return true;
      }
    } else {
      device_id = gpus[0];
      Caffe::SetDevice(gpus[0]);
      Caffe::set_mode(Caffe::GPU);
      return true;
    }
  }
  return false;
}

void AssignCPU() {
  // TFS is a multi-threaded application;
  // avoid using multi-threaded, fork-based OpenBLAS
  // since this is quite likely to cause problems executing
  // caffe::forward(..) within a critical section
  // (and could lead to deadlocks).
  openblas_set_num_threads(1);
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
}

}  // namespace

CaffeSessionOptions::CaffeSessionOptions()
    : force_cpu_only{false},
      force_gpu_id{-1},
      initial_shape{nullptr},
      named_initial_shapes() {}

CaffeServingSession::CaffeServingSession(const CaffeMetaGraphDef& graph,
                                         const CaffeSessionOptions& opts)
    : net_{nullptr}, class_labels_{nullptr}, batch_size_{0}, ts_() {
  int dev_id = ts_.run(
      [](std::unique_ptr<caffe::Net<float>>* net,
         const caffe::NetParameter* graph, const CaffeSessionOptions* opts) {
        int dev_id = -1;
        bool success = !(opts->force_cpu_only)
                           ? TryAssignGPU(opts->force_gpu_id, dev_id)
                           : false;

        if (!success) {
          AssignCPU();
        }

        net->reset(new caffe::Net<float>(*graph));
        return dev_id;
      },
      &net_, &graph.model_def, &opts);

  LOG(INFO) << "Caffe execution mode: "
            << (dev_id >= 0 ? strings::StrCat("GPU (device id: ", dev_id, ")")
                            : "CPU");
  {  // map blob names to indices
    const std::vector<string>& blobs = net_->blob_names();
    for (int idx : net_->input_blob_indices()) {
      input_blob_map_.emplace(blobs[idx], idx);
    }
    for (int idx : net_->output_blob_indices()) {
      output_blob_map_.emplace(blobs[idx], idx);
    }
  }

  {  // blob reshaping
    auto& net_blobs = net_->blobs();
    if (opts.initial_shape) {
      if (input_blob_map_.size() == 1) {
        net_blobs[std::begin(input_blob_map_)->second]->Reshape(
            *opts.initial_shape);
      } else {
        LOG(WARNING) << "Could not reshape input Tensor."
                     << "Network has more than one input.";
      }
    }

    // reshape named input blobs
    for (const auto& kvp : opts.named_initial_shapes) {
      auto it = input_blob_map_.find(kvp.first);
      if (it != input_blob_map_.end()) {
        net_blobs[it->second]->Reshape(kvp.second);
      } else {
        LOG(WARNING) << "Could not reshape Tensor. Input Tensor " << it->first
                     << " does not exist in this network.";
      }
    }
  }

  // class labels
  if (graph.classes.dtype() != DT_INVALID) {
    class_labels_.reset(new Tensor());
    if (!class_labels_->FromProto(graph.classes)) {
      class_labels_.release();
    }
  }

  batch_size_ = BatchSizeOf(*net_);
  LOG(INFO) << "Loaded Network:"
            << "\n  name: " << net_->name()
            << "\n  inputs: " << input_blob_map_.size()
            << "\n  outputs: " << output_blob_map_.size()
            << "\n  initial batch-size: " << batch_size_
            << "\n  output classes: " << (class_labels_ == nullptr
                                              ? "(none)"
                                              : class_labels_->DebugString());
}

CaffeServingSession::~CaffeServingSession() {
  LOG(INFO) << "Unloading Network.";
}

Status CaffeServingSession::Run(
    const std::vector<std::pair<string, Tensor>>& inputs,
    const std::vector<string>& output_tensor_names,
    const std::vector<string>& target_node_names,
    std::vector<Tensor>* outputs) {
  // can't do anything with target_nodes..
  if (target_node_names.size() > 0) {
    return errors::InvalidArgument("target_node_names is not supported by ",
                                   "the Caffe backend");
  }
  // check inputs are present, assuming there are no duplicates
  if (inputs.size() == 0 || inputs.size() < input_blob_map_.size()) {
    return errors::InvalidArgument("Expected ", input_blob_map_.size(),
                                   " inputs, but got ", inputs.size(), ".");
  }
  // determine the batch size from the first input only
  unsigned int batch_size = 0;
  {
    const Tensor& in = inputs[0].second;
    if (in.dims() < 2) {
      return errors::InvalidArgument(
          "Could not determine the batch size; "
          "input must have at least 2 dimensions");
    }
    batch_size = in.dim_size(0);
    if (batch_size < 1) {
      return errors::InvalidArgument("Invalid batch size of ", batch_size);
    }
  }

  outputs->clear();

  if (batch_size_ != batch_size) {
    TF_RETURN_IF_ERROR(Reshape(batch_size));
  }

  return ts_.run(
      [&](caffe::Net<float>* net) {
        // copy inputs to blobs
        auto& net_blobs = net->blobs();
        for (const std::pair<string, Tensor>& in : inputs) {
          auto it = input_blob_map_.find(in.first);
          if (it == input_blob_map_.end()) {
            return errors::InvalidArgument("Input Tensor ", in.first,
                                           " does not exist in the network.");
          } else if (in.second.dim_size(0) != batch_size) {
            return errors::InvalidArgument("Input Tensor ", in.first,
                                           " has an incorrect batch size.");
          }
          // TODO(rayg): validate all other dimensions before copy
          const auto view = in.second.flat<float>();
          unsigned idx = it->second;
          std::copy_n(view.data(), view.size(),
                      net_blobs[idx]->mutable_cpu_data());
        }

        try {
          // execute
          net->Forward();
        } catch (const std::exception& ex) {
          return errors::Internal("Caffe failed to execute the model: ",
                                  ex.what());
        } catch (...) {
          // Boost.python thows a non-std std::exception;
          // attempt to catch it here.
          auto py_err = PythonStatus();
          if (!py_err.ok())
            return py_err;
          else
            return errors::Unknown("Caffe failed to execute the model");
        }

        // copy to output tensors
        for (const string& out : output_tensor_names) {
          if (out == kClassLabelTensorName) {
            // class labels is a special case
            TF_RETURN_IF_ERROR(OutputClassLabels(outputs));
          } else {
            caffe::Blob<float>* blob;
            // search the net for the relevant blob
            auto it = output_blob_map_.find(out);
            if (it == output_blob_map_.end()) {
              // try and find an arbitary blob in the
              // network of the same name
              blob = net->blob_by_name(out).get();
              if (blob == nullptr) {
                return errors::InvalidArgument("Specified network output '",
                                               out, "' does not exist.");
              }
            } else {
              blob = net_blobs[it->second].get();
            }
            const std::vector<int> shape = blob->shape();
            size_t num_elements = shape[0];
            TensorShape out_shape{shape[0]};

            for (size_t k = 1; k < shape.size(); ++k) {
              out_shape.AddDim(shape[k]);
              num_elements *= shape[k];
            }

            Tensor t =
                AsTensor<float>({blob->cpu_data(), num_elements}, out_shape);
            outputs->push_back(t);
          }
        }
        return Status::OK();
      },
      net_.get());
}

Status CaffeServingSession::OutputClassLabels(std::vector<Tensor>* outputs) {
  if (class_labels_ == nullptr) {
    return errors::InvalidArgument(
        "Class labels were requested but none have been loaded");
  }
  outputs->emplace_back(*class_labels_);
  return Status::OK();
}

Status CaffeServingSession::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  caffe::NetParameter param;

  if (!caffe::ReadProtoFromBinaryFile(trained_filename, &param)) {
    return errors::InvalidArgument(strings::StrCat(
        "Caffe network failed to load pretrained layers from file: ",
        trained_filename));
  }

  return ts_.run(
      [](caffe::Net<float>* net, const caffe::NetParameter& param) {
        net->CopyTrainedLayersFrom(param);
        return Status::OK();
      },
      net_.get(), param);
}

Status CaffeServingSession::Reshape(unsigned int batch_size) {
  if (batch_size <= 0) {
    return errors::InvalidArgument("batch_size must be at least 1");
  }
  if (batch_size_ == batch_size) {
    return Status::OK();
  }

  batch_size_ = ts_.run(
      [](caffe::Net<float>* net, unsigned int batch_size) {
        for (int idx : net->input_blob_indices()) {
          auto& blob = *(net->blobs().at(idx));
          std::vector<int> new_shape{blob.shape()};

          if (new_shape.size() > 1 && new_shape[0] > 0) {
            new_shape[0] = batch_size;
            blob.Reshape(new_shape);
          }
        }
        net->Reshape();
        return batch_size;
      },
      net_.get(), batch_size);

  LOG(INFO) << "Reshaped Network (batch_size=" << batch_size_ << ").";
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow

namespace caffe {

::tensorflow::Status ResolveNetInsOuts(const caffe::NetParameter& in_param,
                                       std::vector<string>& in_blobs,
                                       std::vector<string>& out_blobs) {
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  NetParameter filtered_param;
  caffe::Net<float>::FilterNet(in_param, &filtered_param);

  // Create a copy of filtered_param with splits
  // added where necessary.
  NetParameter param;
  InsertSplits(filtered_param, &param);

  std::unordered_set<string> available_blobs;
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    const LayerParameter& lp = param.layer(layer_id);

    // layer inputs
    int num_bottom = lp.bottom_size();
    for (int idx = 0; idx < num_bottom; ++idx) {
      const string& blob_name = lp.bottom(idx);

      if (available_blobs.find(blob_name) == available_blobs.end()) {
        return ::tensorflow::errors::InvalidArgument(
            ::tensorflow::strings::StrCat("Unknown bottom blob '", blob_name,
                                          "' (layer '", lp.name(),
                                          "', bottom index ", idx, ")"));
      } else
        available_blobs.erase(blob_name);
    }

    // layer outputs
    int num_top = lp.top_size();
    for (int idx = 0; idx < num_top; ++idx) {
      const string& blob_name = (num_top > idx) ? lp.top(idx) : "(automatic)";

      available_blobs.insert(blob_name);

      // Collect Input layer tops as Net inputs.
      if (lp.type() == "Input") {
        in_blobs.push_back(blob_name);
      }
    }
  }

  // remaining blobs are outputs
  for (auto it = available_blobs.begin(); it != available_blobs.end(); ++it)
    out_blobs.push_back(*it);

  return ::tensorflow::Status::OK();
}

}  // namespace caffe