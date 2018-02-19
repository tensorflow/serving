#ifndef TENSORFLOW_SERVING_SERVABLES_PROVIDER_MODEL_SELECTOR_H_
#define TENSORFLOW_SERVING_SERVABLES_PROVIDER_MODEL_SELECTOR_H_

#include "tensorflow_serving/servables/selector/model_selector.pb.h"
#include "tensorflow_serving/apis/model.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

class ModelSelector {
public:
  ModelSelector(float sum_weights, const std::vector<Candidate>& candidates, const google::protobuf::Map<std::string, SignatureGroup>& signature_groups)
          : sum_weights(sum_weights), candidates(candidates), signature_groups(signature_groups) {}
  ~ModelSelector();
  Status GetModelSpec(const std::string& signature_def, ModelSpec* model_sepc) const;
private:
  const float sum_weights;
  const std::vector<Candidate> candidates;
  const google::protobuf::Map<std::string, SignatureGroup> signature_groups;
};

} // namespace serving
} // namespace tensorflow

#endif // TENSORFLOW_SERVING_SERVABLES_PROVIDER_MODEL_SELECTOR_H_
