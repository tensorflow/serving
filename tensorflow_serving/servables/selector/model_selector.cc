#include <random>

#include "tensorflow_serving/servables/selector/model_selector.h"
#include "tensorflow_serving/servables/tensorflow/util.h"

namespace tensorflow {
namespace serving {

Status ModelSelector::GetModelSpec(const std::string& signature_def, ModelSpec* model_spec) const {
  static thread_local std::mt19937 mt;
  std::uniform_real_distribution<float> rand(0.0, sum_weights);
  auto r = rand(mt);
  for (auto& candidate : candidates) {
    r -= candidate.weight();
    if (r <= 0) {
      auto name = candidate.name();
      auto group_iter = signature_groups.find(signature_def);
      if (group_iter == signature_groups.end()) {
        return errors::Internal("Cannot find signatureGroup of ", signature_def);
      }
      auto signatures = group_iter->second.signatures();
      auto signature_iter = signatures.find(name);
      if (signature_iter == signatures.end()) {
        return errors::Internal("Cannot find signature_def of ", name);
      }
      MakeModelSpec(name, signature_iter->second,
                    candidate.has_version() ? make_optional(candidate.version().value()) : nullopt, model_spec);
      return Status::OK();
    }
  }
  return errors::Internal("Something wrong.");
}

} // namespace serving
} // namespace tensorflow
