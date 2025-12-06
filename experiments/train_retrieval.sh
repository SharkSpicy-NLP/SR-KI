#!/bin/bash

# Config
KB_SIZE=1000
top_k_kb_train=100
kb_token_layer_frequency=1 # default is 1

retrieval_layers=24 # 24 for Qwen2.5-7B-Instruct

# Model & Data Config
model_name_or_path="Your/Base/Trained/Model/Path"

# Task Proportions
task_proportions='{
    "single_entityQA": 0.4,
    "multi_entityQA_same": 0.2,
    "multi_entityQA_diff": 0.2,
    "unanswerable": 0.2
}'

model_name=$(basename ${model_name_or_path})
echo "model_name=${model_name}"

project_type="linear"
dataset_path="../wikidata/wikiqa_train_150000-format.jsonl"
eval_datapath="../wikidata/wikiqa_eval_2000-format.jsonl"
save_total_limit=5

dataset_basename=$(basename ${dataset_path} .jsonl)

# Train Data Loading
knowledge_key_path="../wikidata/wikidata_train/wikidata_train_bge-large-zh_embd_key.npy"
knowledge_value_path="../wikidata/wikidata_train/wikidata_train_bge-large-zh_embd_value.npy"
material_key_path="../wikidata/wikidata_train/wikidata_train_bge-large-zh_embd_material_key.npy"
material_value_path="../wikidata/wikidata_train/wikidata_train_bge-large-zh_upletters_embd_material_value.npy"
wiki_idx_map_path="../wikidata/wikidata_train/wikidata_train_idx_map.json"
identical_prefix_map_path="../wikidata/wikidata_train/wikidata_train_identical_prefix_map.json"

# Eval Data Loading
eval_knowledge_key_path="../wikidata/wikidata_eval/wikidata_eval_bge-large-zh_embd_key.npy"
eval_knowledge_value_path="../wikidata/wikidata_eval/wikidata_eval_bge-large-zh_embd_value.npy"
eval_material_key_path="../wikidata/wikidata_eval/wikidata_eval_bge-large-zh_embd_material_key.npy"
eval_material_value_path="../wikidata/wikidata_eval/wikidata_eval_bge-large-zh_upletters_embd_material_value.npy"
eval_wiki_idx_map_path="../wikidata/wikidata_eval/wikidata_eval_idx_map.json"
eval_identical_prefix_map_path="../wikidata/wikidata_eval/wikidata_eval_identical_prefix_map.json"

# Train Config
num_train_epochs=5
per_device_train_batch_size=10
gradient_accumulation_steps=5
warmup_ratio=0.01
weight_decay=0.0001
lr_scheduler_type="cosine"
lr=0.0001
max_seq_len=245
save_steps=150
eval_steps=150

output_dir="./train_results/SR-KI_Retrieval_Training_Output"

deepspeed="./ds_config.json"

# Log filepath
LOG_FILENAME="./SR-KI_Retrieval_Training.log"

nohup deepspeed --master_port=9902 --include localhost:0 train.py \
    --retrieval_layers ${retrieval_layers} \
    --return_retr_logits \
    --top_k_kb_train ${top_k_kb_train} \
    --task_proportions "$task_proportions" \
    --do_eval \
    --sep_query_head \
    --model_name_or_path ${model_name_or_path} \
    --kb_token_layer_frequency ${kb_token_layer_frequency} \
    --kb_size ${KB_SIZE} \
    --dataset_path ${dataset_path} \
    --knowledge_key_path ${knowledge_key_path} \
    --knowledge_value_path ${knowledge_value_path} \
    --material_key_path ${material_key_path} \
    --material_value_path ${material_value_path} \
    --wiki_idx_map_path ${wiki_idx_map_path} \
    --identical_prefix_map_path ${identical_prefix_map_path} \
    --eval_steps ${eval_steps} \
    --eval_datapath ${eval_datapath} \
    --eval_knowledge_key_path ${eval_knowledge_key_path} \
    --eval_knowledge_value_path ${eval_knowledge_value_path} \
    --eval_material_key_path ${eval_material_key_path} \
    --eval_material_value_path ${eval_material_value_path} \
    --eval_wiki_idx_map_path ${eval_wiki_idx_map_path} \
    --eval_identical_prefix_map_path ${eval_identical_prefix_map_path} \
    --save_total_limit ${save_total_limit} \
    --project_type ${project_type} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps}  \
    --output_dir ${output_dir} \
    --deepspeed ${deepspeed} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_ratio ${warmup_ratio} \
    --weight_decay ${weight_decay} \
    --save_steps ${save_steps} \
    --lr ${lr} \
    --max_seq_len ${max_seq_len} \
    > ${LOG_FILENAME} 2>&1 &
    
    
    