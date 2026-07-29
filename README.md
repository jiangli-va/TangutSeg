# TangutSeg 西夏文分词研究

## 项目概述

TangutSeg 是一个面向**西夏文自动分词**的研究项目，也是已知**首个**系统研究西夏文词界识别和词性标注的工作。

西夏文（Tangut script）是公元11世纪创制的表意文字，用于书写已灭绝的党项羌语言。西夏文连续书写，词与词之间无显式分隔符，且不再有母语者能够参与标注。本项目由中国社会科学院西夏文专家逐句标注的 2,750 句语料（31,893 词次）作为监督信号，同时引入两类外部资源：**传统西夏文辞书**（约 19,000 个多字词条）和**四行对译材料中的无标注西夏文文本**（约 36 万字符），在极低资源条件下构建分词系统。

由于西夏文数据稀缺且宝贵，标注语料成本高昂，本项目目前没有公开全部训练数据的计划，但提供了少量的语料示例(`corpus_example/`)。同时，我们将开源所有训练代码和最优模型，供研究者使用。

本项目受中国社会科学院学科建设“登峰战略”资助计划（DF2023TS05）、中国社会科学院语言学重点实验室（2024SYZH001）资助。

请参考：
```txt
```

### 核心思路

项目将分词统一建模为字符级 BIES 序列标注，从三个层次利用异质资源：

1. **词典格网**（Lexicon Grid）：不采用单一最大匹配，而是保留所有竞争候选词，编码为 20 维字符级词典特征（BIE×词长 + 可靠度 + 元数据）。词条可靠度通过贝叶斯平滑根据与训练语料的一致性校准，OOF（Out-of-Fold）方式避免过拟合。
2. **无标注语料统计**（Unlabeled Stats）：从约 36 万字符的无标注西夏文文本中提取词频、字符关联度（dPMI / Dice / t-score）和边界熵等分布特征，作为 CRF 或 BiLSTM-CRF 的额外输入。
3. **上下文预训练**（TangutEncoder）：轻量级 4 层 Transformer，通过 MLM 预训练学习西夏文字符的上下文表示，下游接 BIES-CRF 解码。

### 主要结果（5 折交叉验证）

| 模型 | F1 | OOV-R | 说明 |
|:---|---:|---:|:---|
| CRF (baseline) | ~0.88 | ~0.32 | 仅字符 n-gram 模板特征 |
| CRF+BIE+rel_all (dict_core) | 0.9002 | 0.4632 | 17维词典格网（BIE+可靠度） |
| CRF+dict_full | 0.9024 | 0.4690 | 20维，含 has_yi/has_yin/has_book_title |
| CRF+dict_full+dist_all | ~0.905 | ~0.487 | + 全部 gap 分布特征 (最优 CRF) |
| TangutEncoder (MLM) | ~0.90 | **更高** | 预训练对 OOV 召回更优 |

词典格网和 gap 统计特征在极小标注规模下效果显著；预训练 TangutEncoder 在总体性能相当的同时，对未登录词和跨领域文本表现出更强的泛化能力。


## 项目结构

```txt
TangutSeg\
├── README.md
├── corpus_example\         # 语料示例
│   ├── jingshu.txt         # 标注经书语料示例
│   ├── shisu.txt           # 标注世俗文献语料示例
│   ├── dict_example.json   # 西夏文结构化词典示例
│   └── sihang.txt          # 无标注文本示例
├── data\                   # 数据处理脚本
├── models\                 # 分词模型训练程序
├── evaluation\             # 评测脚本
├── pretrain\               # 预训练模型训练程序
├── tag_model\              # 词性标注模型训练程序
├── utils\                  # 工具脚本
├── config.py               # 配置文件
├── run_seg.py              # 分词主程序
├── run_tag.py              # 词性标注主程序
├── run_pretrain.py         # 预训练模型训练主程序
└── requirements.txt
```

## 运行方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练基础分词模型(`crf` or `bilstm_crf`)

```bash
python run_seg.py
```

- 运行参数说明
```bash
--max 300    # 截断训练语料前300条，用于快速调试
--cv         # 启用K折交叉验证，默认5折
--folds 10   # 指定交叉验证折数
--methods ....  # 指定训练方法，默认全部，多个方法用逗号隔开
          dict  # 最大匹配所有方法
          dict1 # 最大匹配法，词典来源于训练语料
          dict2 # 最大匹配法，词典来源于外部词典
          dict3 # 最大匹配法，词典来源于训练语料+外部词典
          crf   # CRF模型所有方法
          crf0  # CRF基线模型，不使用额外特征
          crf1  # CRF + BIE × 词长(11维)
          crf2  # CRF + BIE + Relation Seen(14维)
          crf3  # CRF + Relation All(17维)
          crf4  # CRF + BIE + Meta(词典特征，11+3维)
          crf5  # CRF + dict_all(20维)
          crf6  # CRF + dict_all + freq(20+2维)
          crf7  # CRF + dict_all + freq + 关联度量(20+2+2维)
          crf8  # CRF + dict_all + dist_all(20+2+2+4维)
          crf9  # CRF + dict_all + entropy(20+2+4维)
          bilstm  # BiLSTM-CRF模型所有方法
          bilstm0 # BiLSTM-CRF基线模型，不使用额外特征
          bilstm1 # BiLSTM-CRF + BIE × 词长(11维)
          bilstm2 # BiLSTM-CRF + BIE + Relation Seen(14维)
          bilstm3 # BiLSTM-CRF + Relation All(17维)
          bilstm4 # BiLSTM-CRF + rel_all + internal_dict(17 + 11维)
          bilstm5 # BiLSTM-CRF + rel_all + internal_dict + domain(17 + 11 + 2维)
          bilstm6 #     + freq(17 + 11 + 2 + 2维)
          bilstm7 #     + freq + 关联度量(17 + 11 + 2 + 4维)
          bilstm8 #     + freq + 关联度量 + entropy(17 + 11 + 2 + 8维)
```

### 训练分词-词性标注联合模型

```bash
python run_tag.py
```
- 参数说明

```bash
--max 300    # 截断训练语料前300条，用于快速调试
--cv         # 启用K折交叉验证，默认5折
--folds 10   # 指定交叉验证折数
--methods ....         # 指定训练方法，默认全部，多个方法用逗号隔开
          crf_joint0   # CRF联合标注baseline，无外部特征
          crf_joint5   # CRF + 20维词典特征
          crf_joint6   # CRF + 20维词典特征 + 2维freq
          crf_joint7   # CRF + 20维词典特征 + 2维freq + 2维关联度量
          crf_joint8   # CRF + 20维词典特征 + 8维分布特征
          bilstm_joint0 # BiLSTM-CRF联合标注baseline，无外部特征
          bilstm_joint5 # BiLSTM-CRF + 20维词典特征
          bilstm_joint6 # BiLSTM-CRF + 20维词典特征 + 2维freq
          bilstm_joint7 # BiLSTM-CRF + 20维词典特征 + 2维freq + 2维关联度量
          bilstm_joint8 # BiLSTM-CRF + 20维词典特征 + 8维分布特征
```

### 训练预训练模型(`Transformer-Random`, `Transformer-Char2Vec` or `TangutEncoder`)

```bash
python run_pretrain.py
```
- 参数说明

```bash
--train-w2v    # 训练Char2Vec模型并保存
--pretrain     # 进行MLM预训练TangutEncoder模型
    --masked-mode <mixed(默认)/single>  # 指定MLM掩码模式，mixed为混合掩码，single为单字掩码
    --max-steps <int>  # 预训练最大步数，默认5000
    --batch-size <int> # 预训练批量大小，默认32
--phase2       # 对TangutEncoder进行第二阶段的词感知预训练，需要先进行上一阶段训练
    --lambda-word <float> # 词损失权重，默认0.3
    --lr-encoder <float>  # 编码器学习率，默认5e-5
    --lr-span-head <float> # span head学习率，默认3e-4
    --phase2-max-steps <int> # 第二阶段预训练最大步数，默认3000
    --word-warmup-steps <int> # 词损失warmup步数，默认500
    --num-neg-per-pos <int> # 每个正样本对应的负样本数量，默认5
    --eval-interval <int> # 评估间隔步数，默认200
    --early-stop-patience <int> # 早停耐心值，默认5
    --grad-clip <float> # 梯度裁剪阈值，默认1.0
    --weight-decay <float> # 权重衰减，默认0.01
    --phase2-output-dir <path> # 第二阶段预训练输出目录，默认"output/pretrain/tangut_encoder_phase2"
--seg          # 运行下游分词任务评估
    --cv       # 启用K折交叉验证，默认5折
    --folds <int> # 指定交叉验证折数
    --max <int> # 截断训练语料前max条，用于快速调试
    --methods <list> # 指定训练方法，默认全部，多个方法用逗号隔开
        TEnc-Random   # 随机Transformer + BIES-CRF
        TEnc-Char2Vec # Word2Vec 字符向量初始化 + BIES-CRF
        TEnc-Random+dict  # 随机Transformer + 词典特征
        TEnc-MLM+dict     # MLM预训练Transformer + 词典特征
        TEnc-MLM+dict+gap # MLM预训练Transformer + 词典特征 + gap特征
        CRF-U # 对照当前最优CRF模型（dict_all + dist_all）
    --w2v-model <path> # 指定预训练的Char2Vec模型路径
--pretrained-model <path>   # 指定预训练模型，直接做下游评估
```
