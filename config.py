"""全局配置中心 —— 所有路径、参数集中管理，方便切换语言/语料/方法。"""
from pathlib import Path

# ======================== 基础路径 ========================
BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "corpus" / "all_cleaned.txt"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# ======================== 数据划分 ========================
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1         # test = 1.0 - train - dev
RANDOM_SEED = 42
MAX_SENTENCES = None      # None=全部；int=截断（快速调试用，如5000）

# ======================== 词典法参数 ========================
DICT_MATCH_MODE = "bidirectional"   # "fmm" | "bmm" | "bidirectional"

# ======================== CRF 参数 ========================
CRF_PARAMS = {
    "c1": 1.0,                     # L1 正则化系数
    "c2": 1e-3,                    # L2 正则化系数
    "max_iterations": 200,
    "all_possible_transitions": True,
}

# ======================== BiLSTM-CRF 参数 ========================
BILSTM_CRF_PARAMS = {
    "embedding_dim": 100,          # 词向量维度
    "hidden_dim": 64,             # LSTM 隐层维度
    "num_layers": 2,               # LSTM 层数
    "dropout": 0.5,                # Dropout 率
    "learning_rate": 5e-4,         # 学习率
    "batch_size": 512,              # 批量大小
    "epochs": 10000,                # 训练轮数（极大值，实际由早停控制）
    "device": "auto",              # "auto" | "cpu" | "cuda" | "cuda:0"
    "lr_patience": 5,              # LR 调度: 开发集 loss 不降则衰减
    "lr_factor": 0.3,              # LR 衰减因子
    "early_stop_patience": 3,       # 早停: 验证集 loss 连续不降次数
    "grad_clip": 5.0,              # 梯度裁剪
    "dict_dropout": 0.2,           # 训练时对词典特征块做 dropout (连续特征)
    "domain_dist_dim": 2,          # 领域分布向量维度 (经书比例, 世俗比例)
    "jingshu_loss_weight": 1.0,     # 经书句子损失权重 (>1.0 加强经书, =1.0 不变)
}

# ======================== 外部工具 ========================
TOOL_PATHS = {
    "ltp": "LTP",                   # 或自定义模型路径
    "hanlp": None,                  # None = 自动加载默认模型
    "stanza": "zh",
    "trankit": "chinese",
}

# ======================== 无标注语料 (gap 特征) ========================
UNLABELED_JSON_PATH = BASE_DIR / "corpus" / "提取四行对译中的西夏字（以典籍的图片为单位）.json"

# 关联度量: "dpmi" = 折扣PMI (默认, 效果最好), "dice" = Dice系数, "t_score" = t-score
# 控制 UnlabeledStatsExtractor 的特征列 1 和 5 (L_assoc / R_assoc) 使用哪种度量
BIGRAM_METRIC = "dpmi"

# ======================== 日志 ========================
VERBOSE = True
