#!/bin/bash

# Model path mapping
declare -A MODEL_PATH_DICT=(
  ["bge-m3"]="../../models/bge-m3"
  ["bge-large-zh"]="../../models/bge-large-zh-v1.5"
)

# Model name to use
model_name="bge-large-zh"

# Get model path from mapping
model_path="${MODEL_PATH_DICT[$model_name]}"

if [ -z "$model_path" ]; then
  echo "Error: Model $model_name not found in MODEL_PATH_DICT"
  exit 1
fi

# Define dataset configuration array
declare -a dataset_paths=(
  "../wikidata/wikiqa_train_150000-format.jsonl"
  "../wikidata/wikiqa_eval_2000-format.jsonl"
  "../wikidata/wikiqa_test_20000-format.jsonl"
)

declare -a dataset_names=(
  "wikidata_train"
  "wikidata_eval"
  "wikidata_test"
)

# Generate KB embeddings for each dataset
for i in "${!dataset_paths[@]}"; do
  echo "Processing dataset: ${dataset_names[$i]}"
  echo "Dataset path: ${dataset_paths[$i]}"
  echo "Model path: ${model_path}"
  
  CUDA_VISIBLE_DEVICES=0 python generate_kb_wiki.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --dataset_name "${dataset_names[$i]}" \
    --dataset_path "${dataset_paths[$i]}" \
    --output_path ../wikidata/${dataset_names[$i]}
  
  echo "Completed: ${dataset_names[$i]}"
  echo "----------------------------------------"
done