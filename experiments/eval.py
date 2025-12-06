import torch
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.qwen2_5_model import SRKIQwen2ForCausalLM
import numpy as np
from train import _load_cached_embeddings
import torch
import json
import re
import random
import argparse
from tqdm import tqdm
from bert_score import score, BERTScorer
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_Q_qwen2(Q: str, tokenizer):
    messges = [
        {"role": "user", "content": Q},
    ]
    prompt = tokenizer.apply_chat_template(
        messges,
        tokenize=False,
        add_generation_prompt=True,
        add_generation_prompt_token=False,
    )
    return prompt

def extract_field(model_output, idx, qa_type, h, r, t=None):
    match = re.search(fr"{re.escape(h)}的{re.escape(r)}是(.+?)[\[\s，,。]", model_output)
    return match.group(1) if match else ""

def caculate_acc(A: str, outputs_text: str, qa_type:str):
    def get_docid_letter(text: str): 
        matches = re.findall(r'\[([A-Z])\]', text)
        if len(matches) == 0:
            return None
        return matches
    
    if qa_type == "unanswerable":
        return 1 if A.strip() == outputs_text.strip() else 0

    A = get_docid_letter(A)
    outputs_text = get_docid_letter(outputs_text)
    correct_count = 0
    if A is None or outputs_text is None:
        return 0
    if len(A) != len(outputs_text): # if cannot get the ID, return 0 directly
        return 0
    for a, o in zip(A, outputs_text):
        if a==o:
            correct_count += 1
    return correct_count / len(A)

def caculate_bertscore(data, outputs_text, scorer: BERTScorer):
    nodes = data["nodes"]
    qa_type = data["qa_type"]
    ex_pred = []
    if qa_type == "single_entityQA":
        heads = [nodes[0]] 
        relations = [nodes[1]] 
        tails = [nodes[2]]
    elif qa_type in ["multi_entityQA_diff", "multi_entityQA_same"]:
        heads = [nodes[0], nodes[3]]
        relations = [nodes[1], nodes[4]]
        tails = [nodes[2], nodes[5]]
    elif qa_type == "unanswerable": # unanswerable directly evaluate the answer text and the output score
        P, R, F1 = scorer.score([outputs_text], [data["A_format"]])
        return {
                "P": P.item(),
                "R": R.item(),
                "F1": F1.item()
            }
    else:
        raise ValueError(f"error qa_type {qa_type}")

    for idx, (h, r) in enumerate(zip(heads, relations)):
        ex_pred.append(extract_field(outputs_text, idx, qa_type, h, r, tails[idx]))

    total_P = []
    total_R = []
    total_F1 = []
    for e_p, t in zip(ex_pred, tails):
        P, R, F1 = scorer.score([e_p], [t])
        total_P.append(P.item())
        total_R.append(R.item())
        total_F1.append(F1.item())
    
    return {
        "P": np.mean(total_P),
        "R": np.mean(total_R),
        "F1": np.mean(total_F1)
    }

    
def get_related_key_value_embds(
        data, 
        knowledge_key_embds, 
        knowledge_value_embds,
        material_key_embds,
        material_value_embds,
        kb_size, 
        wiki_idx_map=None, 
        identical_prefix_map=None,
    ):  
    if kb_size == 0:
        return None, None
    wiki_id = [d["wiki_id"] for d in data["kb_info"]]
    doc_id = [ord(d) - ord('A') for d in data["doc_id"]]
    prefix_wiki_id = ["-".join(i.split("-")[:2]) for i in wiki_id]
    context_size = (kb_size-2)//2 if data["qa_type"] in ["single_entityQA"] else (kb_size - 4)//2 
    
    target_idx = [wiki_idx_map[i] for i in wiki_id] # [len(wiki_id)]
    key_embeds_list = []
    value_embeds_list = []
    kb_set_len = len(knowledge_key_embds)
    
    if data["qa_type"] != "unanswerable":
        train_knowledge_key = knowledge_key_embds[target_idx] # [len(wiki_id), out_dim]
        train_knowledge_val = knowledge_value_embds[target_idx] # [len(wiki_id), out_dim]
    
        train_material_set_key = material_key_embds[target_idx] # [len(doc_id), out_dim]
        train_material_set_val = material_value_embds[doc_id] # [len(doc_id), out_dim]
        
        key_embeds_list.extend([train_knowledge_key, train_material_set_key])
        value_embeds_list.extend([train_knowledge_val, train_material_set_val])
    else:
        context_size = kb_size // 2
    
    identical_prefix_idx = []
    for i in prefix_wiki_id:
        identical_prefix_idx += identical_prefix_map[i]
    mask = np.ones(kb_set_len, dtype=bool)
    mask[list(set(target_idx + identical_prefix_idx))] = False
    context_scope = np.where(mask)[0]
    context_indices = np.random.choice(context_scope, context_size, replace=False)

    context_know_key = knowledge_key_embds[context_indices] # [context_set_size, out_dim]
    context_know_value = knowledge_value_embds[context_indices] # [context_set_size, out_dim]
    context_material_key = material_key_embds[context_indices] # [context_set_size, out_dim]
    context_material_value = material_value_embds[np.random.choice(list(range(26)), context_size, replace=True)]
    
    key_embeds_list.extend([context_know_key, context_material_key])
    value_embeds_list.extend([context_know_value, context_material_value])
            
    new_key_embds = torch.concat(key_embeds_list, 0)
    new_value_embds = torch.concat(value_embeds_list, 0)
    
    answer_idx = []
    if args.random_pos:
        indices = torch.randperm(new_key_embds.shape[0])
        new_key_embds = new_key_embds[indices]
        new_value_embds = new_value_embds[indices]
        logger.info(f"debug: random pos: {indices[:10]}")
        
        if data["qa_type"] == "single_entityQA":
            for i in [0, 1]:
                answer_idx.append(torch.where(indices==i)[0].item())
        else:
            for i in [0, 1, 2, 3]:
                answer_idx.append(torch.where(indices==i)[0].item())
    else:
        if data["qa_type"] == "single_entityQA":
            answer_idx = [0, 1]
        else:
            answer_idx = [0, 1, 2, 3]
    return new_key_embds, new_value_embds, answer_idx

def statistics(eval_type, results):
    need_append = []
    if "acc" in eval_type:
        acc_res = [r["metric"]["acc"] for r in results if r["qa_type"] != "unanswerable"]
        correct_num = sum(acc_res)
        acc = round(correct_num / len(acc_res), 4)
        need_append.append({
            "correct": correct_num,
            "total": len(acc_res),
            "acc": acc
        })
        logger.info(f"acc_res: {acc}")
    if "bertscore" in eval_type:
        total_metric = [r["metric"]["bertscore"] for r in results]
        bertscore_res = {}
        for metric in total_metric:
            for k, v in metric.items():
                if k not in bertscore_res:
                    bertscore_res[k] = []
                bertscore_res[k].append(v)
        for k, v in bertscore_res.items():
            bertscore_res[k] = np.mean(v)
        need_append.append({
            "bertscore": bertscore_res
        })
        logger.info(f"bertscore_res: {json.dumps(bertscore_res, ensure_ascii=False, indent=2)}")
    if "kb_recall" in results[0]["metric"]:
        kb_recall = [r["metric"]["kb_recall"] for r in results if r["qa_type"] != "unanswerable"]
        kb_all_res = defaultdict(list)
        for k_r in kb_recall:
            for k, v in k_r.items():
                kb_all_res[k].append(v)
        for k, v in kb_all_res.items():
            kb_all_res[k] = np.mean(v)
        need_append.append(kb_all_res)
        logger.info(f"kb_recall: {json.dumps(kb_all_res, ensure_ascii=False, indent=2)}")

    results.append(need_append)
    
    return results

def sample_dataset(dataset, need_num):
    singleQA = [d for d in dataset if d["qa_type"] == "single_entityQA"]
    multi_entityQA_same = [d for d in dataset if d["qa_type"] == "multi_entityQA_same"]
    multi_entityQA_diff = [d for d in dataset if d["qa_type"] == "multi_entityQA_diff"]
    unanswerable = [d for d in dataset if d["qa_type"] == "unanswerable"]
    
    sample_singleQA = random.sample(singleQA, int(need_num*TASK_PROPORTION["single_entityQA"]))
    sample_multi_entityQA_same = random.sample(multi_entityQA_same, int(need_num*TASK_PROPORTION["multi_entityQA_same"]))
    sample_multi_entityQA_diff = random.sample(multi_entityQA_diff, int(need_num*TASK_PROPORTION["multi_entityQA_diff"]))
    sample_unanswerable = random.sample(unanswerable, int(need_num*TASK_PROPORTION["unanswerable"]))
    
    sampled_dataset = sample_singleQA + sample_multi_entityQA_same + sample_multi_entityQA_diff + sample_unanswerable
    logger.info(f"Total sampled {len(sampled_dataset)} data")
    return sampled_dataset


def main(
        model_name_or_path: str,
        knowledge_key_path: str,
        knowledge_value_path: str,
        material_key_path: str,
        material_value_path: str,
        wiki_idx_map_path: str,
        identical_prefix_map_path: str,
        dataset_path: str,
        output_dir: str,
        kb_size: int,
        test_num: int,
):
    # Set seed
    set_seed(args.seed)
    # Load embeddings
    knowledge_key_embds, knowledge_value_embds, _ = _load_cached_embeddings(knowledge_key_path, knowledge_value_path)
    material_key_embds, material_value_embds, _ = _load_cached_embeddings(material_key_path, material_value_path)

    wiki_idx_map = json.load(open(wiki_idx_map_path, "r", encoding="utf-8"))
    identical_prefix_map = json.load(open(identical_prefix_map_path, "r", encoding="utf-8"))
    
    if args.save_attention:
        attention_save_loc=os.path.join(output_dir, "attention_weights")
        if not os.path.exists(attention_save_loc):
            os.makedirs(attention_save_loc, exist_ok=True)
    
    # Load config
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    model_config.return_retr_logits = False
    model_config.tokenizer_path = model_name_or_path
    
    if args.top_k_kb is not None:
        if getattr(model_config, "knowledge_info", None) is not None:
            model_config.knowledge_info = model_config.knowledge_info
        else:
            model_config.knowledge_info = [24]
            logger.info(f"Unknown Knowledge Info for {model_name_or_path}, set to [24] default")
        model_config.dynamic_sparsify = True
        model_config.top_k_kb = args.top_k_kb
    else:
        model_config.dynamic_sparsify = False
    
    # Set output_path
    output_filename = f"{'-'.join(args.eval_type)}_results-{test_num}.json"

    if args.top_k_kb is not None:
        output_filename = output_filename.replace(".json", f"-topk-{args.top_k_kb}.json")
    
    if args.reuse_kb:
        output_filename = output_filename.replace(".json", "-reuse_kb.json")
        model_config.reuse_kb = True
    else:
        model_config.reuse_kb = False

    output_path = os.path.join(output_dir, output_filename)
    
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    model = SRKIQwen2ForCausalLM.from_pretrained(
        model_name_or_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    
    # Load dataset
    dataset = open(dataset_path, "r", encoding="utf-8").readlines()
    dataset = [json.loads(line) for line in dataset]
    
    dataset = sample_dataset(dataset, test_num)
    results = []
    
    if "bertscore" in args.eval_type:
        bert_scorer = BERTScorer(lang="zh", model_type=BERT_SCORER_PATH, num_layers=12, rescale_with_baseline=True, device=model.device)
    
    for idx, data in enumerate(tqdm(dataset)):
        qa_type = data["qa_type"]
        Q = data["Q_format"]  
        A = data["A_format"]
      
        related_key_embds, related_value_embds, answer_idx = get_related_key_value_embds(data, knowledge_key_embds, knowledge_value_embds, material_key_embds, material_value_embds, kb_size, wiki_idx_map, identical_prefix_map)
        
        if related_key_embds is not None:
            related_key_embds = related_key_embds.to(model.device)
            related_value_embds = related_value_embds.to(model.device)
        input_kb = {"key_embds": related_key_embds, "value_embds": related_value_embds}
        
        prompt = format_Q_qwen2(Q, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs_len = len(inputs["input_ids"][0])
        inputs = {k:v.to(model.device) for k, v in inputs.items()}
        
        # Save attention weights
        if args.save_attention:
            attn_save_path = os.path.join(attention_save_loc, f"{str(idx)}_{qa_type}")
            if not os.path.exists(attn_save_path):
                os.makedirs(attn_save_path, exist_ok=True)
            save_attention_config = {
                "save_attention_weights": True,
                "attention_save_loc": attn_save_path,
                "attention_file_base_name": os.path.basename(model_name_or_path)
            }
        else:
            save_attention_config = {}
        
        model.config.answer_idx = answer_idx
        kb_recall = {}
        for key in list(vars(model.config).keys()):
            if '@' in key and "recall" in key:
                logger.info(f"del {key}")
                delattr(model.config, key)
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False, 
            use_cache=True, 
            **input_kb,
            **save_attention_config
        )
        outputs_text = tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)
        
        for k_l in model.knowledge_info:
            kb_recall[f"recall_{k_l}@top"] = getattr(model.config, f"recall_{k_l}@top", 0)
            kb_recall[f"recall_{k_l}@10"] = getattr(model.config, f"recall_{k_l}@10", 0)
            kb_recall[f"recall_{k_l}@100"] = getattr(model.config, f"recall_{k_l}@100", 0)
            kb_recall[f"recall_{k_l}@{args.top_k_kb}"] = getattr(model.config, f"recall_{k_l}@{args.top_k_kb}", 0)
        
        metric = {"kb_recall": kb_recall}

        if "acc" in args.eval_type:
            metric = {"acc": caculate_acc(A, outputs_text, data["qa_type"]), **metric}     
        if "bertscore" in args.eval_type:
            metric = {"bertscore": caculate_bertscore(data, outputs_text, bert_scorer), **metric}
            
        logger.info(f"Q: {Q}")
        logger.info(f"A: {A}")
        logger.info(f"outputs_text: {outputs_text}")
        logger.info(f"metric: {json.dumps(metric, ensure_ascii=False, indent=2)}")
        logger.info("-"*100)

        temp_res = {
            "Q": Q,
            "A": A,
            "outputs_text": outputs_text,
            "qa_type": data["qa_type"],
            "metric": metric
        }
            
        results.append(temp_res)
        if idx % 50 == 0:
            json.dump(results, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    
    results = statistics(eval_type=args.eval_type, results=results)  
    json.dump(results, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--knowledge_key_path", type=str, default="")
    parser.add_argument("--knowledge_value_path", type=str, default="")
    parser.add_argument("--material_key_path", type=str, default="")
    parser.add_argument("--material_value_path", type=str, default="")
    parser.add_argument("--wiki_idx_map", type=str, default="")
    parser.add_argument("--identical_prefix_map", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--kb_size", type=int, default=0)
    parser.add_argument("--eval_type", type=str, nargs='+')
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--top_k_kb", type=int, default=None)
    parser.add_argument("--save_attention", action="store_true", default=False)
    parser.add_argument("--random_pos", action="store_true", default=False, help="randomly shuffle the positions of the injected KB")
    parser.add_argument("--reuse_kb", action="store_true", default=False, help="reuse KB after retrieval layer")
    parser.add_argument("--seed", type=int, default=42, help="seed for testing")
    parser.add_argument("--task_proportions", type=str, default=None, help="Task proportions")
    parser.add_argument("--bert_scorer_path", type=str, default=None, help="BERT scorer path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    TASK_PROPORTION = json.loads(args.task_proportions)
    BERT_SCORER_PATH = args.bert_scorer_path
    logger.info(f"task_proportions: {TASK_PROPORTION}")
    
    main(
        args.model_name_or_path,
        args.knowledge_key_path,
        args.knowledge_value_path,
        args.material_key_path,
        args.material_value_path,
        args.wiki_idx_map,
        args.identical_prefix_map,
        args.dataset_path,
        args.output_dir, 
        args.kb_size,
        args.test_num,
    )