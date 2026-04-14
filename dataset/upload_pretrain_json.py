# from huggingface_hub import HfApi

# api = HfApi()
# repo_id = "susuahi/seq-monkey"
# file_path = "blockdata/dataset/seq_monkey_pretrain.jsonl"

# # 1. 创建数据集仓库（不存在则创建）
# api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

# # 2. 直接上传jsonl文件
# api.upload_file(
#     path_or_fileobj=file_path,
#     path_in_repo="data/seq_monkey_pretrain.jsonl",  # 仓库内路径
#     repo_id=repo_id,
#     repo_type="dataset",
#     token="my-token"
# )

# print("✅ 原始JSONL上传完成！")


# import os
# import tempfile
# from datasets import load_dataset

# # 1. 配置（国内必加）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# # 你的文件路径
# input_jsonl = "blockdata/dataset/seq_monkey_pretrain.jsonl"
# repo_id = "susuahi/seq-monkey"  # 替换成你的仓库ID

# # 2. 加载数据（关闭streaming，用临时目录缓存）
# # 关键：指定临时缓存目录，用完自动删
# with tempfile.TemporaryDirectory() as tmp_cache:
#     dataset = load_dataset(
#         "json",
#         data_files=input_jsonl,
#         split="train",
#         streaming=False,  # 关闭流式
#         num_proc=1,        # 单进程，避免并行报错
#         cache_dir=tmp_cache # 临时缓存，自动清理
#     )

#     # 3. 上传到 Hugging Face
#     dataset.push_to_hub(
#         repo_id,
#         max_shard_size="50MB",
#         private=True,  # 设为False则公开
#         token="my-token"  # 或提前huggingface-cli login
#     )

# print("✅ 上传完成！")


# import os
# from datasets import load_dataset

# # 1. 配置（国内镜像 + 挂载盘缓存路径）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# # 挂载盘路径（你的大容量磁盘，绝对够用）
# CACHE_DIR = "/blockdata/hf_cache"

# # 你的文件配置
# input_jsonl = "blockdata/dataset/seq_monkey_pretrain.jsonl"
# repo_id = "susuahi/seq-monkey"

# # 2. 自动创建挂载盘上的缓存文件夹
# os.makedirs(CACHE_DIR, exist_ok=True)

# # 3. 加载数据（缓存存在大容量挂载盘，不占系统盘）
# dataset = load_dataset(
#     "json",
#     data_files=input_jsonl,
#     split="train",
#     num_proc=1,
#     cache_dir=CACHE_DIR  # 🔥 核心：缓存写在挂载盘，不是系统盘
# )

# # 4. 上传到 Hugging Face
# dataset.push_to_hub(
#     repo_id,
#     max_shard_size="50MB",
#     private=True,
#     token="my-token"
# )

# # 5. 上传完成后，清理缓存（可选，释放空间）
# import shutil
# shutil.rmtree(CACHE_DIR)

# print("✅ 上传完成！缓存已自动清理！")

import os
import shutil
from datasets import load_dataset

# ==============================================
# 🔥 核心：把所有临时/缓存路径，全改到数据盘 /blockdata
# ==============================================
# 1. 国内镜像（必须）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 强制HF所有缓存到数据盘（覆盖默认的/root/.cache/huggingface）
os.environ["HF_HOME"] = "hf_home"
os.environ["HF_DATASETS_CACHE"] = "hf_datasets_cache"

# 3. 强制/tmp临时目录到数据盘（解决最大元凶！所有程序临时文件都写这里）
DATA_TMP = "tmp"
os.makedirs(DATA_TMP, exist_ok=True)
os.environ["TMPDIR"] = DATA_TMP
os.environ["TEMP"] = DATA_TMP
os.environ["TMP"] = DATA_TMP

# ==============================================
# 业务配置（不用改）
# ==============================================
CACHE_DIR = "hf_cache"  # 数据集缓存
input_jsonl = "dataset/seq_monkey_pretrain.jsonl"
repo_id = "susuahi/seq-monkey"
HF_TOKEN = "my-token"

# ==============================================
# 执行流程
# ==============================================
# 1. 创建所有需要的目录
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# 2. 加载数据集（缓存/临时文件全在数据盘，不占系统盘）
print("🔧 开始加载数据集（缓存/临时文件全在数据盘，不占系统盘）...")
dataset = load_dataset(
    "json",
    data_files=input_jsonl,
    split="train",
    num_proc=1,
    cache_dir=CACHE_DIR
)

# 3. 上传到Hugging Face（所有临时分片、缓存全在数据盘）
print("☁️ 开始上传到Hugging Face...")
dataset.push_to_hub(
    repo_id,
    max_shard_size="50MB",
    private=True,
    token=HF_TOKEN
)

# 4. 上传完成后，自动清理所有临时/缓存文件（释放数据盘空间）
print("🧹 开始清理缓存/临时文件...")
shutil.rmtree(CACHE_DIR, ignore_errors=True)
shutil.rmtree(DATA_TMP, ignore_errors=True)
shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)
shutil.rmtree(os.environ["HF_DATASETS_CACHE"], ignore_errors=True)

print("✅ 上传完成！系统盘零占用，所有缓存已自动清理！")