import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import re
from typing import List
import os
import torch.distributed as dist
import string
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoConfig, set_seed, TrainingArguments, Trainer, SchedulerType
import random
import deepspeed
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qwen2_5_model import SRKIQwen2ForCausalLM
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils.tensor_fragment import fragment_address 
from numpy._core.multiarray import _reconstruct
from numpy import ndarray, dtype, uint32
from numpy.dtypes import UInt32DType
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.serialization.add_safe_globals([LossScaler, ZeroStageEnum, fragment_address, _reconstruct, ndarray, dtype, UInt32DType, uint32])

# Load cached embeddings
def _load_cached_embeddings(key_path:str, value_path:str):
    key_embds = np.load(key_path)
    value_embds = np.load(value_path)
    key_embds = torch.from_numpy(key_embds).to(torch.bfloat16)
    value_embds = torch.from_numpy(value_embds).to(torch.bfloat16)
    if len(key_embds.shape) == 1:
        key_embds = key_embds.unsqueeze(0)
        value_embds = value_embds.unsqueeze(0)
    output_dim = key_embds.shape[1]
    logger.info(f"Loading cached embeddings, key_embds shape:{key_embds.shape}, value_embds shape:{value_embds.shape}")
    return key_embds, value_embds, output_dim

# Get trainable parameters
def _get_trainable_parameters(
    model: SRKIQwen2ForCausalLM,
    kb_token_layer_frequency: int,
    sep_query_head: bool,
    need_init: bool = True,
):
    llm_q_params = []
    llm_q_param_names = []
    param_names = ["k_proj_new", "v_proj_new"] if not sep_query_head else ["q_proj_new", "k_proj_new", "v_proj_new"]
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        layer_id = re.search(r"\d+", name)
        if not layer_id:
            continue
        layer_id = int(layer_id[0])
        for param_name in param_names:
            if param_name in name and layer_id % kb_token_layer_frequency == 0:
                param.requires_grad = True
                if "new" in param_name and "q_proj_new" not in param_name and need_init:
                    if param.dim() >= 2:
                        torch.nn.init.xavier_normal_(param)
                    else:
                        torch.nn.init.zeros_(param)

                if "q_proj_new" in param_name and need_init:
                    param.data.copy_(model.get_parameter(name.replace("q_proj_new", "q_proj")).data)
                    
                llm_q_params.append(param)
                llm_q_param_names.append(name)

    return llm_q_params


# Get format answer
def get_format_answer(d, qa_type):
    nodes = d["nodes"]
    doc_id = d["doc_id"]
    if "single_entityQA" in qa_type:
        d["A_format"] = f"根据提供的物料库信息，{nodes[0]}的{nodes[1]}是{nodes[2]}[{doc_id[0]}]"
    elif "multi_entityQA_same" in qa_type:
        d["A_format"] = f"根据提供的物料库信息，{nodes[0]}的{nodes[1]}是{nodes[2]}[{doc_id[0]}]，{nodes[4]}是{nodes[5]}[{doc_id[1]}]"
    elif "multi_entityQA_diff" in qa_type:
        d["A_format"] = f"根据提供的物料库信息，{nodes[0]}的{nodes[1]}是{nodes[2]}[{doc_id[0]}]，{nodes[3]}的{nodes[4]}是{nodes[5]}[{doc_id[1]}]"
    return d


class TrainDataset(Dataset):
    def __init__(self, 
                 tokenizer, 
                 data_path, 
                 max_seq_len: int, 
                 knowledge_key_embds, 
                 knowledge_value_embds, 
                 kb_size_config, 
                 material_key_embds=None, 
                 letter_embeds=None,
                 wiki_idx_map=None, 
                 identical_prefix_map=None):
        super(TrainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.dataset = open(data_path, "r", encoding="utf-8").readlines()
        self.dataset = [json.loads(d) for d in self.dataset]
        self.max_seq_len = max_seq_len 
        self.kb_set_len = len(knowledge_key_embds)
        self.qa_type_map = {
            "single_entityQA":0,
            "multi_entityQA_diff":1,
            "multi_entityQA_same":1,
            "unanswerable":2
        }
        
        self.knowledge_key_embds = knowledge_key_embds
        self.knowledge_value_embds = knowledge_value_embds
        self.material_key_embds = material_key_embds
        self.material_value_embds = letter_embeds # [26, out_dim]
        self.wiki_idx_map = wiki_idx_map
        self.identical_prefix_map = identical_prefix_map
        self.kb_size = kb_size_config
    

    def get_key_embeddings(self, data, kb_size, wiki_idx_map=None, identical_prefix_map=None):     
        wiki_id = [d["wiki_id"] for d in data["kb_info"]]
        doc_id = [ord(d) - ord('A') for d in data["doc_id"]]
        prefix_wiki_id = ["-".join(i.split("-")[:2]) for i in wiki_id]       
        context_size = (kb_size-2)//2 if data["qa_type"] in ["single_entityQA"] else (kb_size - 4)//2 
        train_idx = [wiki_idx_map[i] for i in wiki_id] # [len(wiki_id)]
        key_embeds_list = []
        value_embeds_list = []
        
        if data["qa_type"] != "unanswerable":
            train_knowledge_key = self.knowledge_key_embds[train_idx] # [len(wiki_id), out_dim]
            train_knowledge_val = self.knowledge_value_embds[train_idx] # [len(wiki_id), out_dim]
        
            train_material_set_key = self.material_key_embds[train_idx] # [len(doc_id), out_dim]
            train_material_set_val = self.material_value_embds[doc_id] # [len(doc_id), out_dim]
            
            key_embeds_list.extend([train_knowledge_key, train_material_set_key])
            value_embeds_list.extend([train_knowledge_val, train_material_set_val])
        else:
            context_size = kb_size // 2
                    
        identical_prefix_idx = []
        for i in prefix_wiki_id:
            identical_prefix_idx += identical_prefix_map[i]
        mask = np.ones(self.kb_set_len, dtype=bool)
        mask[list(set(train_idx + identical_prefix_idx))] = False
        context_scope = np.where(mask)[0]
        context_indices = np.random.choice(context_scope, context_size, replace=False)

        context_know_key = self.knowledge_key_embds[context_indices] # [context_set_size, out_dim]
        context_know_value = self.knowledge_value_embds[context_indices] # [context_set_size, out_dim]
        context_material_key = self.material_key_embds[context_indices] # [context_set_size, out_dim]
        context_material_value = self.material_value_embds[np.random.choice(list(range(26)), context_size, replace=True)] # [context_set_size, out_dim]
        
        key_embeds_list.extend([context_know_key, context_material_key])
        value_embeds_list.extend([context_know_value, context_material_value])
               
        new_key_embds = torch.concat(key_embeds_list, 0)
        new_value_embds = torch.concat(value_embeds_list, 0)
            
        return new_key_embds, new_value_embds
        
    def update_kb_size(self, new_kb_size):
        self.kb_size = max(0, new_kb_size) 
        
    def _format_QA(self, Q: str, A: str):
        messages = [
            {"role": "user", "content": Q},
            {"role": "assistant", "content": A}
        ]

        messages_Q = [
            {"role": "user", "content": Q}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, add_generation_prompt_token=False)
        prompt_Q = self.tokenizer.apply_chat_template(messages_Q, tokenize=False, add_generation_prompt=True, add_generation_prompt_token=False)

        return prompt, prompt_Q
    
    def _create_single_label(self, input_str: str, q_input: str):
        q = self.tokenizer(q_input, return_tensors="pt", padding=False, max_length=self.max_seq_len, truncation=True)["input_ids"][0]
        q_len = len(q)
        label = self.tokenizer(input_str, return_tensors="pt", padding=False, max_length=self.max_seq_len, truncation=True)["input_ids"][0]
        label[:q_len] = -100
        
        if len(label) < self.max_seq_len:
            if self.tokenizer.padding_side == "left":
                label = torch.cat([torch.tensor([-100] * (self.max_seq_len - len(label)), dtype=label.dtype), label])
            else:
                label = torch.cat([label, torch.tensor([-100] * (self.max_seq_len - len(label)), dtype=label.dtype)])
        return label

    def get_qa_type(self, idx):
        return self.dataset[idx]["qa_type"]

    def __len__(self):
        return len(self.dataset)
    
    def get_random_letter(self, length=2):
        return random.choices(string.ascii_uppercase, k=length)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data["doc_id"] = self.get_random_letter(len(data["doc_id"]))
        data = get_format_answer(data, data["qa_type"])
        
        input_str, q_input = self._format_QA(data["Q_format"], data["A_format"])
        
        tokenizer_output = self.tokenizer(input_str, return_tensors="pt", padding="max_length", max_length=self.max_seq_len, truncation=True)
        input_ids = tokenizer_output["input_ids"][0]
        attention_mask = tokenizer_output["attention_mask"][0]
        
        labels = self._create_single_label(input_str, q_input)
        
        qa_types = self.qa_type_map[data["qa_type"]]
        key_embds, value_embds = self.get_key_embeddings(
            data=data,
            kb_size=self.kb_size,
            wiki_idx_map=self.wiki_idx_map, 
            identical_prefix_map=self.identical_prefix_map,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "key_embds": key_embds,
            "value_embds": value_embds,
            "qa_types": qa_types
        }


# Customized training Sampler
class KBDataSampler(Sampler):
    def __init__(self, dataset: Dataset, proportions: dict, batch_size: int, drop_last=False):
        logger.info("KBDataSampler.__init__ starts...")
        self.dataset = dataset
        self.proportions = proportions
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        self.qa_type2idxs = {}
        logger.info("Analyzing QA types in the dataset...")
        for idx in range(len(self.dataset)):
            qa_type = self.dataset.get_qa_type(idx)
            self.qa_type2idxs.setdefault(qa_type, []).append(idx)
        logger.info("QA types analysis completed")
        
        assert set(proportions.keys()) == set(self.qa_type2idxs.keys()), "QA types proportion is not consistent with the QA types in the dataset"
        
        for idxs in self.qa_type2idxs.values():
            random.shuffle(idxs)
        
        self.pos = {tp: 0 for tp in self.qa_type2idxs}
    
        logger.info("KBDataSampler.__init__ completed")
        
    def __iter__(self):
        batch_buffer = []
        produced_samples = 0
        epoch_samples = len(self.dataset)

        while produced_samples < epoch_samples:
            if not batch_buffer:
                batch_buffer.extend(self._next_batch())
            idx = batch_buffer.pop(0)
            produced_samples += 1
            yield idx
                
    def _next_batch(self):
        batch = []
        for tp, frac in self.proportions.items():
            n = max(int(round(frac * self.batch_size)), 1)
            idx_list = self.qa_type2idxs[tp]
            start = self.pos[tp]
            end = start + n
            if end > len(idx_list):
                random.shuffle(idx_list)
                start = 0
                end = n
            batch.extend(idx_list[start:end])
            self.pos[tp] = end
        while len(batch) < self.batch_size:
            tp = random.choice(list(self.proportions.keys()))
            idx_list = self.qa_type2idxs[tp]
            idx = idx_list[self.pos[tp] % len(idx_list)]
            self.pos[tp] += 1
            batch.append(idx)
            
        return batch
    
    def __len__(self):
        return len(self.dataset)

class ShardSampler(Sampler):
    def __init__(self, base_sampler):
        assert dist.is_initialized(), "Distributed environment not initialized"
        self.base_sampler = base_sampler
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

    def __iter__(self):
        for i, idx in enumerate(self.base_sampler):
            if i % self.num_replicas == self.rank:
                yield idx

    def __len__(self):
        return (len(self.base_sampler) + self.num_replicas - 1) // self.num_replicas
    
# Build Trainer
class DynamicKBTrainer(Trainer):
    
    def __init__(self, kb_size_config=None, task_proportions=None, *args, **kwargs):
        logger.info("DynamicKBTrainer.__init__ starts...")
        super().__init__(*args, **kwargs)
        self.kb_size_config = kb_size_config
        self.current_step = 0
        self.proportions = task_proportions
        logger.info("DynamicKBTrainer.__init__ completed")
            
    def compute_loss(self, model, inputs, return_outputs=False):
        key_embds = inputs.get("key_embds")
        value_embds = inputs.get("value_embds")
        if self.current_step % self.args.logging_steps == 0:
            logger.info(f"Step {self.current_step}: Using key_embds = {key_embds.shape}, value_embds = {value_embds.shape}")
        return super().compute_loss(model, inputs, return_outputs)

    def get_train_dataloader(self):
        logger.info("Creating training data loader...")
        logger.info(f"Training dataset size: {len(self.train_dataset)}")
        logger.info(f"Batch size: {self.args.per_device_train_batch_size}")
        logger.info(f"Task proportions: {self.proportions}")
        
        sampler = KBDataSampler(self.train_dataset, self.proportions, self.args.per_device_train_batch_size)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            sampler = ShardSampler(sampler)
        logger.info(f"Sampler created, total samples: {len(sampler)}")
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            persistent_workers=self.args.dataloader_persistent_workers,
            pin_memory=True
        )
        logger.info("Training data loader created")
        return dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen with DeepSpeed‑ZeRO3 via Trainer")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-7B", help="HF repo or local path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to json/jsonl dataset")
    parser.add_argument("--eval_datapath", type=str, default=None, help="Path to eval json/jsonl dataset")
    parser.add_argument("--wiki_idx_map_path", type=str, default=None, help="Path to wiki idx map")
    parser.add_argument("--identical_prefix_map_path", type=str, default=None, help="Path to identical prefix map")
    parser.add_argument("--eval_wiki_idx_map_path", type=str, default=None, help="Path to eval wiki idx map")
    parser.add_argument("--eval_identical_prefix_map_path", type=str, default=None, help="Path to eval identical prefix map")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save checkpoints")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config; will auto‑create if missing")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume from checkpoint")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Use gradient checkpointing")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local GPU rank, do not manually pass")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--project_type", type=str, default="linear", help="Projector type", choices=["linear", "mlp"])
    parser.add_argument("--project_mlp_hidden_dim", type=int, default=7168, help="Project MLP hidden dim")
    parser.add_argument("--do_eval", action="store_true", default=False, help="Do evaluation")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Save total limit")
    parser.add_argument("--need_init", action="store_true", default=False, help="Need initialize")
    parser.add_argument("--top_k_kb_train", type=int, default=None, help="Top k KB for training")
    parser.add_argument("--knowledge_key_path", type=str, default=None, help="Path to knowledge key embeddings")
    parser.add_argument("--knowledge_value_path", type=str, default=None, help="Path to knowledge value embeddings")
    parser.add_argument("--material_key_path", type=str, default=None, help="Path to material key embeddings")
    parser.add_argument("--material_value_path", type=str, default=None, help="Path to material value embeddings")
    parser.add_argument("--eval_knowledge_key_path", type=str, default=None, help="Path to eval knowledge key embeddings")
    parser.add_argument("--eval_knowledge_value_path", type=str, default=None, help="Path to eval knowledge value embeddings")
    parser.add_argument("--eval_material_key_path", type=str, default=None, help="Path to eval material key embeddings")
    parser.add_argument("--eval_material_value_path", type=str, default=None, help="Path to eval material value embeddings")
    parser.add_argument("--kb_size", type=int, default=20, help="KB size")
    parser.add_argument("--kb_token_layer_frequency", type=int, default=10, help="KB token injection frequency, inject every k layers")
    parser.add_argument("--top_k_kb", type=int, default=20, help="Top k KB")
    parser.add_argument("--dynamic_sparsify", action="store_true", default=False, help="Dynamic sparsify")
    parser.add_argument("--sep_query_head", action="store_true", default=False, help="Separate query head")
    parser.add_argument("--return_retr_logits", action="store_true", default=False, help="Return retrieval logits")
    parser.add_argument("--task_proportions", type=str, default=None, help="Task proportions")
    parser.add_argument("--retrieval_layers", type=int, nargs="+", default=None, help="Retrieval layers")
    return parser.parse_args()


def seed_everything(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)
    deepspeed.init_distributed()
    
    # Load embeddings
    knowledge_key_path = args.knowledge_key_path
    knowledge_value_path = args.knowledge_value_path
    material_key_path = args.material_key_path
    material_value_path = args.material_value_path
    knowledge_key_embds, knowledge_value_embds, embed_dim = _load_cached_embeddings(knowledge_key_path, knowledge_value_path)
    material_key_embds, material_value_embds, embed_dim = _load_cached_embeddings(material_key_path, material_value_path)
    
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
        
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model_config.kb_layer_frequency = args.kb_token_layer_frequency 
    model_config.embed_dim = embed_dim
    model_config.top_k_kb = args.top_k_kb
    model_config.dynamic_sparsify = args.dynamic_sparsify
    model_config.sep_query_head = args.sep_query_head
    model_config.project_type = args.project_type
    model_config.return_retr_logits = args.return_retr_logits
    
    if args.top_k_kb_train:
        model_config.dynamic_sparsify = True
        model_config.top_k_kb = args.top_k_kb_train
    if args.project_type == "mlp":
        projector_kwargs = {"mlp_hidden_dim": args.project_mlp_hidden_dim}
        model_config.projector_kwargs = projector_kwargs
            
    task_proportions = json.loads(args.task_proportions)
    logger.info("task_proportions: ", task_proportions)
    
    if args.return_retr_logits:
        # Retrieval layer information
        KNOWLEDGE_INFO = args.retrieval_layers
        assert len(KNOWLEDGE_INFO) == 1, "Only one retrieval layer is supported"
        model_config.knowledge_info = KNOWLEDGE_INFO
        retrieval_layer_weights = list(range(model_config.num_hidden_layers))
        for layer_idx in KNOWLEDGE_INFO:
            retrieval_layer_weights[layer_idx] = 1.0 / len(KNOWLEDGE_INFO)
        model_config.retrieval_layer_weights = retrieval_layer_weights
    
    model = SRKIQwen2ForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
  
    model.train()
    kb_size_config = args.kb_size

    wiki_idx_map = json.load(open(args.wiki_idx_map_path, "r", encoding="utf-8"))
    identical_prefix_map = json.load(open(args.identical_prefix_map_path, "r", encoding="utf-8"))
    train_dataset = TrainDataset(tokenizer, args.dataset_path, args.max_seq_len, knowledge_key_embds, knowledge_value_embds, kb_size_config, material_key_embds, material_value_embds, wiki_idx_map, identical_prefix_map)
    
    if args.do_eval:
        eval_knowledge_key_embds, eval_knowledge_value_embds, eval_embed_dim = _load_cached_embeddings(args.eval_knowledge_key_path, args.eval_knowledge_value_path)
        eval_material_key_embds, eval_material_value_embds, eval_embed_dim = _load_cached_embeddings(args.eval_material_key_path, args.eval_material_value_path)
        eval_wiki_idx_map = json.load(open(args.eval_wiki_idx_map_path, "r", encoding="utf-8"))
        eval_identical_prefix_map = json.load(open(args.eval_identical_prefix_map_path, "r", encoding="utf-8"))
        eval_dataset = TrainDataset(tokenizer, args.eval_datapath, args.max_seq_len, eval_knowledge_key_embds, eval_knowledge_value_embds, args.kb_size, eval_material_key_embds, eval_material_value_embds, eval_wiki_idx_map, eval_identical_prefix_map)

    need_train_params = _get_trainable_parameters(model, args.kb_token_layer_frequency, args.sep_query_head, args.need_init)

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name} {param.requires_grad}")

    logger.info("Creating output directory...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory created")

    logger.info("Configuring eval parameters...")
    if args.do_eval:
        eval_args = {
            "do_eval": args.do_eval,
            "evaluation_strategy": "steps",
            "eval_steps": args.eval_steps, 
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
    else:
        eval_args = {}
    logger.info("Eval parameters configured")

    logger.info("Creating training parameters...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy = "steps",
        do_train=True,
        **eval_args,
        save_steps=args.save_steps,
        bf16=True,
        learning_rate=args.lr,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=8,
        dataloader_persistent_workers=False,
        logging_steps=args.logging_steps,
        report_to='tensorboard',
        deepspeed=args.deepspeed,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        save_total_limit=args.save_total_limit,
    )
    logger.info("Training parameters created")

    logger.info("Creating DynamicKBTrainer...")
    trainer = DynamicKBTrainer(
        kb_size_config=kb_size_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        task_proportions=task_proportions
    )
    logger.info("DynamicKBTrainer created")

    if args.resume_from_checkpoint:
        del model
        torch.cuda.empty_cache()
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    
if __name__ == "__main__":
    main()






