## udpated fineweb to help downloading

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "data/edu_fineweb10B"
remote_name = "sample-10BT"

shard_size = int(5e7)  # 🔥 REDUCED from 100M → 50M (important)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint16)

def write_datafile(filename, tokens_list):
    np.save(filename, np.concatenate(tokens_list))

# 🔥 NO multiprocessing (this was killing your memory)
shard_index = 0
tokens_buffer = []
token_count = 0

progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

for doc in fw:
    tokens = tokenize(doc)

    if token_count + len(tokens) < shard_size:
        tokens_buffer.append(tokens)
        token_count += len(tokens)
        progress_bar.update(len(tokens))
    else:
        # write shard
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")

        write_datafile(filename, tokens_buffer)

        shard_index += 1
        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

        # start new shard with leftover tokens
        tokens_buffer = [tokens]
        token_count = len(tokens)

        progress_bar.update(len(tokens))

# final shard
if tokens_buffer:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile(filename, tokens_buffer)