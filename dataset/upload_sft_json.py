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
input_jsonl = "blockdata/dataset/BelleGroup/train_3.5M_CN.json"
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