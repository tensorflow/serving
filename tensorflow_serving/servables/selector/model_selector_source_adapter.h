#ifndef TENSORFLOW_SERVING_SERVABLES_PROVIDER_MODEL_SELECTOR_SOURCE_ADAPTER_H_
#define TENSORFLOW_SERVING_SERVABLES_PROVIDER_MODEL_SELECTOR_SOURCE_ADAPTER_H_

#include "tensorflow_serving/core/storage_path.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/servables/selector/model_selector.h"
#include "tensorflow_serving/servables/selector/model_selector_source_adapter.pb.h"

namespace tensorflow {
namespace serving {

class ModelSelectorSourceAdapter final
        : public SimpleLoaderSourceAdapter<StoragePath, ModelSelector> {
public:
  explicit ModelSelectorSourceAdapter(const ModelSelectorSourceAdapterConfig& config);
  ~ModelSelectorSourceAdapter() override;

private:
  TF_DISALLOW_COPY_AND_ASSIGN(ModelSelectorSourceAdapter);
};

} // namespace serving
} // namespace tensorflow

#endif // TENSORFLOW_SERVING_SERVABLES_PROVIDER_MODEL_SELECTOR_SOURCE_ADAPTER_H_
