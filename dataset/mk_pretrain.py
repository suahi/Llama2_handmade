import pandas as pd
import psutil
from datasets import load_dataset
from tqdm import tqdm
import json
import sys

# def split_text(text, chunk_size=512):
#     """将文本按指定长度切分成块"""
#     return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# data_files="blockdata/dataset/mobvoi_seq_monkey_general_open_corpus.jsonl"
output_pretrain_data="blockdata/dataset/seq_monkey_pretrain.jsonl"

# pubmed_dataset_streamed = load_dataset(
#     "json", 
#     data_files=data_files, 
#     streaming=True
# )

# train_stream = pubmed_dataset_streamed["train"]
# # 使用示例
# with open(output_pretrain_data,'a', encoding='utf-8') as pretrain:
#     for i, example in tqdm(enumerate(train_stream),file=sys.stdout, dynamic_ncols=True):
#         text = example['text']
#         chunks = split_text(text)
#         for chunk in chunks:
#             pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')


dataset = load_dataset("json", data_files=output_pretrain_data, split="train", streaming=True)
dataset.push_to_hub("seq-monkey", max_shard_size="50MB")