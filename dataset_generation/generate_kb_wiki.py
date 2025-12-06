import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import string

def compute_embeddings(
    encoder_model_spec: str, dataset, batch_size: int = 100
) -> np.array:
    """Compute embeddings for the given dataset in batches using the encoder model spec."""
    embeddings_keys = []
    embeddings_values = []
    embeddings_material_keys = []
    all_elements_keys = []
    all_elements_values = []
    all_elements_material_keys = []
    
    all_wiki_ids = set()
    upper_letters = list(string.ascii_uppercase)

    # Build mapping tables
    wiki_idx_map = {}
    identical_prefix_map = {}
    
    for entity in dataset:
        kb_info = entity["kb_info"]
        for info in kb_info:
            wiki_id = info["wiki_id"]
            if wiki_id in all_wiki_ids:
                continue
            
            wiki_idx_map[wiki_id] = len(wiki_idx_map)
            prefix_wiki_id = "-".join(wiki_id.split("-")[:2])
            identical_prefix_map.setdefault(prefix_wiki_id, []).append(wiki_idx_map[wiki_id])
            
            all_wiki_ids.add(wiki_id)
            
            all_elements_keys.append(info["knowledge"]["key"])
            all_elements_values.append(info["knowledge"]["value"])
            all_elements_material_keys.append(info["doc_id"]["key"])
    
    json.dump(wiki_idx_map, open(f"{args.output_path}/{args.dataset_name}_idx_map.json", "w", encoding="utf-8"))
    json.dump(identical_prefix_map, open(f"{args.output_path}/{args.dataset_name}_identical_prefix_map.json", "w", encoding="utf-8"))

    chunks_keys = [
        all_elements_keys[i : i + batch_size]
        for i in range(0, len(all_elements_keys), batch_size)
    ]
    chunks_values = [
        all_elements_values[i : i + batch_size]
        for i in range(0, len(all_elements_values), batch_size)
    ]
    chunks_material_keys = [
        all_elements_material_keys[i : i + batch_size]
        for i in range(0, len(all_elements_material_keys), batch_size)
    ]
    
    model = SentenceTransformer(encoder_model_spec, device="cuda")
    assert len(chunks_keys) == len(chunks_values) == len(chunks_material_keys)
    for chunk_keys, chunk_values, chunk_material_keys in tqdm(zip(chunks_keys, chunks_values, chunks_material_keys), total=len(chunks_keys)):
        embd_keys = model.encode(chunk_keys, convert_to_numpy=True)
        embd_values = model.encode(chunk_values, convert_to_numpy=True)
        embd_material_keys = model.encode(chunk_material_keys, convert_to_numpy=True)

        embeddings_keys.append(embd_keys)
        embeddings_values.append(embd_values)
        embeddings_material_keys.append(embd_material_keys)
    
    upper_letters_embeds = model.encode(upper_letters, convert_to_numpy=True) # [26, embed_dim]

    embeddings_keys = np.concatenate(embeddings_keys, 0)
    embeddings_values = np.concatenate(embeddings_values, 0)
    embeddings_material_keys = np.concatenate(embeddings_material_keys, 0)
    
    assert len(embeddings_keys) == len(all_elements_keys)
    assert len(embeddings_values) == len(all_elements_values)
    assert len(embeddings_material_keys) == len(all_elements_material_keys)
    
    return embeddings_keys, embeddings_values, embeddings_material_keys, upper_letters_embeds

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bge-large-zh",
        choices=["bge-m3", "bge-large-zh"],
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model. If not provided, will use model_name to look up in MODEL_PATH_DICT.",
    )
    parser.add_argument("--dataset_name", type=str, default="wikidata")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Path to the dataset in JSON format.",
    )
    parser.add_argument("--output_path", type=str, default="dataset")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    
    assert args.model_path is not None, "Model path is required."
    model_path = args.model_path
    
    dataset = open(args.dataset_path, "r", encoding="utf-8").readlines()
    dataset = [json.loads(d) for d in dataset]
    
    print(f"Dataset Example: {dataset[0]}")
    print(f"Using model path: {model_path}")
    os.makedirs(args.output_path, exist_ok=True)
    
    key_embeds, value_embeds, material_key_embeds, upper_letters_embeds = compute_embeddings(model_path, dataset)
    save_name = args.model_name

    np.save(
        f"{args.output_path}/{args.dataset_name}_{save_name}_embd_key.npy",
        np.array(key_embeds),
    )
    np.save(
        f"{args.output_path}/{args.dataset_name}_{save_name}_embd_value.npy",
        np.array(value_embeds),
    )
    np.save(
        f"{args.output_path}/{args.dataset_name}_{save_name}_embd_material_key.npy",
        np.array(material_key_embeds),
    )
    np.save(
        f"{args.output_path}/{args.dataset_name}_{save_name}_upletters_embd_material_value.npy",
        np.array(upper_letters_embeds),
    )