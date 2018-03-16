#include "tensorflow_serving/servables/selector/model_selector_source_adapter.h"

#include "tensorflow_serving/util/proto_util.h"

namespace tensorflow {
namespace serving {

Status LoadModelSelectorFromFile(const string& path,
                                 const string& file_name,
                                 std::unique_ptr<ModelSelector>* selector) {
  auto config = ReadProtoFromFile<ModelSelectorConfig>(path + "/" + file_name);
  std::vector<Candidate> candidates;
  candidates.reserve(config.candidates_size());
  float sum_weights = 0;
  for (auto& candidate : config.candidates()) {
    sum_weights += candidate.weight();
    candidates.push_back(candidate);
  }
  selector->reset(new ModelSelector(sum_weights, candidates, config.signaturegroups()));
  return Status::OK();
}

ModelSelectorSourceAdapter::ModelSelectorSourceAdapter(
        const ModelSelectorSourceAdapterConfig& config)
        : SimpleLoaderSourceAdapter<StoragePath, ModelSelector>(
        [config](const StoragePath& path, std::unique_ptr<ModelSelector>* selector) {
          return LoadModelSelectorFromFile(path, config.filename(), selector);
        },
        SimpleLoaderSourceAdapter<StoragePath, ModelSelector>::EstimateNoResources()) {}

ModelSelectorSourceAdapter::~ModelSelectorSourceAdapter() { Detach(); }

class ModelSelectorSourceAdapterCreator {
public:
  static Status Create(
          const ModelSelectorSourceAdapterConfig& config,
          std::unique_ptr<SourceAdapter<StoragePath, std::unique_ptr<Loader>>>* adapter) {
    adapter->reset(new ModelSelectorSourceAdapter(config));
    return Status::OK();
  }
};

REGISTER_STORAGE_PATH_SOURCE_ADAPTER(ModelSelectorSourceAdapterCreator, ModelSelectorSourceAdapterConfig);

} // namespace serving
} // namespace tensorlfow
