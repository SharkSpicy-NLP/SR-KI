#!/bin/bash

model_name_or_path="Your/Trained/Model/Path"
bert_scorer_path="Your/BERT/Scorer/Path"

task_proportions='{
    "single_entityQA": 0.4,
    "multi_entityQA_same": 0.2,
    "multi_entityQA_diff": 0.2,
    "unanswerable": 0.2
}'

dirname=$(basename $(dirname ${model_name_or_path}))
basename=$(basename ${model_name_or_path})

dataset_path="../wikidata/wikiqa_test_20000-format.jsonl"
knowledge_key_path="../wikidata/wikidata_test/wikidata_test_bge-large-zh_embd_key.npy"
knowledge_value_path="../wikidata/wikidata_test/wikidata_test_bge-large-zh_embd_value.npy"
material_key_path="../wikidata/wikidata_test/wikidata_test_bge-large-zh_embd_material_key.npy"
material_value_path="../wikidata/wikidata_test/wikidata_test_bge-large-zh_upletters_embd_material_value.npy"
wiki_idx_map="../wikidata/wikidata_test/wikidata_test_idx_map.json"
identical_prefix_map="../wikidata/wikidata_test/wikidata_test_identical_prefix_map.json"

output_dir="./eval_results"
kb_size=100
test_num=100
top_k_kb=100
eval_type=(acc bertscore)

configs=(
    ""
    "--top_k_kb ${top_k_kb}"
    "--top_k_kb ${top_k_kb} --reuse_kb"
)

for config in "${configs[@]}"; do
    echo "Running evaluation with config: ${config}"
    
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --dataset_path ${dataset_path} \
        --bert_scorer_path ${bert_scorer_path} \
        --model_name_or_path ${model_name_or_path} \
        --knowledge_key_path ${knowledge_key_path} \
        --knowledge_value_path ${knowledge_value_path} \
        --material_key_path ${material_key_path} \
        --material_value_path ${material_value_path} \
        --wiki_idx_map ${wiki_idx_map} \
        --identical_prefix_map ${identical_prefix_map} \
        --eval_type ${eval_type[@]} \
        --output_dir ${output_dir}/${dirname}/${basename}/kb-size-for-test-${kb_size} \
        --kb_size ${kb_size} \
        --test_num ${test_num} \
        --task_proportions "${task_proportions}" \
        ${config} \
    
    echo "Completed evaluation with config: ${config}"
    echo "----------------------------------------"
done
