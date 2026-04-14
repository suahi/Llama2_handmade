import pandas as pd
import psutil
from datasets import load_dataset
from tqdm import tqdm
import json
import sys

def convert_message(data):
    """
    将原始数据转换为标准格式
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

# 文件路径
data_files = "blockdata/dataset/BelleGroup/train_3.5M_CN.json"
output_pretrain_data = "blockdata/dataset/BelleGroup/train_3.5M_CN_sft.jsonl"

# 流式加载数据集
pubmed_dataset_streamed = load_dataset(
    "json", 
    data_files=data_files, 
    streaming=True
)

train_stream = pubmed_dataset_streamed["train"]

# 写入文件
with open(output_pretrain_data, 'a', encoding='utf-8') as pretrain:
    for i, example in tqdm(enumerate(train_stream), file=sys.stdout, dynamic_ncols=True):
        # 只打印前2条测试
        # print('example 类型:', type(example), '内容:', example)
        
        # ===================== 修复区 =====================
        # 1. 直接用 example（已经是字典，无需json.loads）
        # 2. 提取conversations
        conversations = example['conversations']
        # 3. 调用转换函数
        message = convert_message(conversations)
        # ==================================================      
        # 后续写入逻辑（解开注释即可）
        pretrain.write(json.dumps({'messages': message}, ensure_ascii=False) + '\n')
