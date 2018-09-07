#!/bin/bash
if [[ $S3_BUCKET ]]; then
    mkdir -p ${MODEL_BASE_PATH}/${MODEL_NAME}
    echo "Syncing with ${S3_BUCKET} to ${MODEL_BASE_PATH}/${MODEL_NAME} ..."
    aws s3 sync $S3_BUCKET ${MODEL_BASE_PATH}/${MODEL_NAME}/
    echo "Sync complete"
fi

tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}