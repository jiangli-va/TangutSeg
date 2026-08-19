# TangutSeg: 西夏文分词研究

TangutSeg 是一个面向**西夏文自动分词**的研究项目，也是已知**首个**系统研究西夏文自动分词和词性标注的工作。

西夏文(Tangut script)创制于公元11世纪的西夏王朝(Xixia)，是一种用于书写西夏语的语素文字。该文字由西夏统治者李元昊命文臣野利仁荣创制，定为“国书”。西夏于1227年被蒙古帝国所灭，西夏文也随之逐渐湮灭无闻。西夏语属汉藏语系的羌语支，西夏人的语言现已灭绝，学术界认为其与现代的羌语和嘉绒语关系最密切。西夏文字形与汉字相仿，目前总计约6000余字，行体方整，笔划繁复。独体字较少，由2个字甚至3、4个字合成一字者居多数。

西夏文连续书写，词与词之间无显式分隔符，且不再有母语者能够参与标注。本项目将中国社会科学院西夏文专家逐句标注的 2,750 句语料（31,893 词次）作为监督信号，同时引入两类外部资源：**传统西夏文辞书**（约 19,000 个多字词条）和**四行对译材料中的无标注西夏文文本**（约 32 万字符），在极低资源条件下构建分词系统。

由于西夏文数据稀缺且宝贵，标注语料成本高昂，本项目目前没有公开全部训练数据的计划，但提供了少量的语料示例(`corpus_example/`)。同时，我们将开源所有训练代码，供研究者参考。未来我们将开源成熟的最优分词模型，供学界使用。

本项目受中国社会科学院学科建设“登峰战略”资助计划（DF2023TS05）、中国社会科学院语言学重点实验室（2024SYZH001）资助。

请参阅：
```txt

```

## 核心思路

本项目将分词统一建模为字符级 BIES 序列标注，从三个层次利用异质资源：

1. **词典知识**（Lexicon Grid）：通过软词典格网、词条可靠度与词典元数据等特征，辅助 CRF 或 BiLSTM-CRF 模型进行分词。
2. **无标注语料知识**（Unlabeled Stats）：从约 32 万字符的无标注西夏文文本中提取词频、字符关联度和边界熵等分布特征，作为 CRF 或 BiLSTM-CRF 的额外输入。
3. **上下文预训练**（TangutEncoder）：轻量级 3 层 Transformer，通过 MLM 预训练学习西夏文字符的上下文表示，下游接 BIES-CRF 解码。

## 核心结果（5 折交叉验证）

| 模型 | F1 | OOV-R | 说明 |
|:---|---:|---:|:---|
| CRF (baseline) | ~0.88 | ~0.43 | CRF基线模型，仅含离散字符特征 |
| CRF+dict_all | ~0.90 | ~0.47 | 20维特征，含词典格网、可靠度、词典元数据 |
| CRF+dict_all+dist_all | ~0.91 | ~0.49 | 20维词典特征+8维分布特征 |
| TangutEncoder (MLM) | ~0.91 | ~0.62 | MLM预训练，含词典与分布特征 |

词典和分布统计特征在极小标注规模下效果显著；预训练 TangutEncoder 在总体性能上与 CRF 对照模型相当，但在未登录词召回上明显更强，表现出更强的泛化能力。


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
├── train_seg.py            # 分词模型训练程序
├── train_tag.py            # 词性标注模型训练程序
├── train_pretrain.py       # 预训练模型训练程序
├── saved_models\           # 保存的最优模型(有待更新)
├── run_seg.py              # 分词主程序
└── requirements.txt
```

## 运行方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练基础分词模型(`CRF` or `BiLSTM-CRF`)

```bash
python train_seg.py
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
python train_tag.py
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
python train_pretrain.py
```
- 参数说明

```bash
--train-w2v  # 训练Char2Vec模型并保存
--pretrain   # 进行MLM预训练TangutEncoder模型
    --masked-mode <string>  # 指定MLM掩码模式，mixed为混合掩码，single为单字掩码(默认)
    --max-steps <int>          # 预训练最大步数，默认5000
    --batch-size <int>         # 预训练批量大小，默认32
--phase2     # 对TangutEncoder进行第二阶段的词感知预训练，需要先进行上一阶段训练
    --lambda-word <float>      # 词损失权重，默认0.3
    --lr-encoder <float>       # 编码器学习率，默认5e-5
    --lr-span-head <float>     # span head学习率，默认3e-4
    --phase2-max-steps <int>   # 第二阶段预训练最大步数，默认3000
    --word-warmup-steps <int>  # 词损失warmup步数，默认500
    --num-neg-per-pos <int>    # 每个正样本对应的负样本数量，默认5
    --eval-interval <int>      # 评估间隔步数，默认200
    --early-stop-patience <int> # 早停耐心值，默认5
    --grad-clip <float>        # 梯度裁剪阈值，默认1.0
    --weight-decay <float>     # 权重衰减，默认0.01
    --phase2-output-dir <path> # 第二阶段预训练输出目录，默认"output/pretrain/tangut_encoder_phase2"
--seg       # 运行下游分词任务评估
    --cv                       # 启用K折交叉验证，默认5折
    --folds <int>              # 指定交叉验证折数
    --max <int>                # 截断训练语料前max条，用于快速调试
    --methods <list>           # 指定训练方法，默认全部，多个方法用逗号隔开
        TEnc-Random            # 随机Transformer + BIES-CRF
        TEnc-Char2Vec          # Word2Vec 字符向量初始化 + BIES-CRF
        TEnc-Random+dict       # 随机Transformer + 词典特征
        TEnc-MLM+dict          # MLM预训练Transformer + 词典特征
        TEnc-MLM+dict+gap      # MLM预训练Transformer + 词典特征 + gap特征
        CRF-U                  # 对照当前最优CRF模型（dict_all + dist_all）
    --w2v-model <path>         # 指定预训练的Char2Vec模型路径
    --save-model <path>        # 保存完整推理模型，默认"saved_models"
--pretrained-model <path> # 指定预训练模型，直接做下游评估
```

## 数据集

### 专家标注语料

标注语料来自两部西夏文文献：

- 经书：《大宝积经》西夏文译本第六十八卷；
- 世俗文献：《类林》，内容涉及多种主题和语言表达。

多位西夏文专家独立进行分析，再通过比较和联合审定形成参考标注。

| 类别 | 片段数 | 词次 | 词型 |
|:---|---:|---:|---:|
| 经书 | 234 | 3,717 | 769 |
| 世俗文献 | 2,516 | 28,176 | 4,046 |
| 合计 | 2,750 | 31,893 | 4,433 |

这里的“片段”是专家根据西夏文献结构划分的文本单位，不等同于语言学意义下的句子。

### 传统辞书

结构化辞书包含词形、拟音、释义和词条属性。清理并去重后，共获得：

- 19,290 个多字候选词；
- 其中双字词17,082个，占88.6%。

辞书覆盖标注语料中83.1%的词型和95.9%的词次，但大量辞书词条没有在标注语料中出现。因此，本项目不将辞书匹配直接视为正确词界，而将其编码为可由模型选择的软特征。

### 无标注西夏文文本

无标注文本来自俄藏西夏文献四行对译材料中的西夏文原文，包括：

- 663条图版记录；
- 46个题名层级单元；
- 7部文献；
- 约31.8万西夏文字符。

该资源主要用于：

- 提取相邻字符频率；
- 计算字符关联度；
- 计算左右邻接熵；
- 训练Char2Vec；
- 预训练TangutEncoder。

实验仅使用其中的西夏文字符序列，不使用译文自动生成分词标签。

## 方法

### 1. 统一分词框架

给定西夏字符序列

```text
c₁ c₂ ... cₙ
```

模型为每个字符预测一个 BIES 标签：

- `B`：多字词首字；
- `I`：多字词内部字符；
- `E`：多字词末字；
- `S`：单字词。

除最大匹配基线外，所有模型均产生 BIES 发射分数，并通过 CRF 进行全局解码。

### 2. 监督分词模型

项目实现了以下基础模型：

- 双向最大匹配；
- Linear CRF；
- BiLSTM–CRF；
- 随机初始化 Transformer–CRF。

Linear CRF 使用字符、上下文字符、相邻二元组、标点及单字成词等局部特征。BiLSTM 和 Transformer 则学习字符的上下文表示。

### 3. 词典格网特征

对于句子中所有能够与辞书词条匹配的跨度，系统不采用单一最大匹配结果，而是保留所有相互竞争的候选，形成软词典格网。

完整词典表示包含20维：

| 特征组 | 维度 | 说明 |
|:---|---:|:---|
| BIE × 词长 | 11 | 候选词中的字符位置及词长分组 |
| 已观察词条可靠度 | 3 | 作为候选词首部、内部和尾部的最大可靠度 |
| 未观察词条先验 | 3 | 未在训练语料出现的词条类别先验 |
| 辞书元数据 | 3 | 是否有“義”“音”或“書名”属性 |
| 合计 | 20 | 完整词典表示 |

词条可靠度根据其在训练语料中作为字符子串出现和与人工词界完全一致的情况估计：

$$
r(w)=
\dfrac{\text{hit}(w)+\kappa p_{g(w)}}
{\text{occ}(w)+\kappa}.
$$

其中，`hit` 表示与人工词界完全一致的次数，`occ` 表示作为子串出现的次数，`p_g` 是对应词长组的先验可靠度。

为防止训练标签泄露，训练集上的可靠度特征通过内部五折折外统计生成；验证集和测试集仅使用当前外层训练集估计的可靠度。

### 4. 显式分布特征

系统从无标注文本中为每个字符左右两侧的空隙提取8维特征：

| 特征组 | 维度 | 说明 |
|:---|---:|:---|
| Freq | 2 | 左、右二元字符频率 |
| Coo | 2 | 字符关联强度 |
| Entropy | 4 | 左右邻接熵 |
| 合计 | 8 | 完整显式分布表示 |

字符关联度支持 dPMI、Dice 和 t-score。邻接熵用于衡量字符与左右邻接字符搭配的自由程度，为潜在词界提供连续证据。

### 5. 表示学习

项目比较三种字符表示：

- `Transformer-Random`：随机初始化 Transformer；
- `Transformer-Char2Vec`：使用无标注文本训练的静态字符向量初始化；
- `TangutEncoder`：通过字符级 MLM 预训练的上下文编码器。

TangutEncoder 的主要配置为：

- 3层 Transformer；
- 隐层维度192；
- 4个注意力头；
- 一个西夏字符对应一个 token；
- 约15%的字符被遮盖；
- 混合单字遮盖与长度为2–4的连续片段遮盖。

下游模型将TangutEncoder输出的上下文表示与词典特征、语料统计特征融合，最后通过BIES–CRF完成分词。

## 评测设置

实验采用按文献类别分层的五折交叉验证。每轮使用一折作为测试集，并从其余数据中划分训练集和验证集。

主要指标包括：

- `P`：词级精确率；
- `R`：词级召回率；
- `F1`：词级严格匹配 F1；
- `OOV-R`：未出现在当前有标注训练词表中的词形召回率；
- `IV-R`：出现在当前有标注训练词表中的词形召回率。

只有预测词的起止位置与人工标注完全一致时才视为正确。

这里的 OOV 仅表示词形未出现在当前折的有标注训练语料中；该词仍可能出现在外部辞书或无标注文本中。当前评测考察同一来源内留出文本的泛化能力，不代表对全新文献的跨文献迁移能力。

## 主要结果

以下为五折交叉验证的代表性结果。表中数值为五折均值与标准差。

| 模型 | P | R | F1 | OOV-R | IV-R |
|:---|---:|---:|---:|---:|---:|
| Dict-corpus | 0.845±0.005 | 0.890±0.003 | 0.867±0.004 | 0.219±0.014 | **0.954±0.002** |
| CRF | 0.876±0.009 | 0.889±0.007 | 0.883±0.008 | 0.432±0.020 | 0.932±0.006 |
| BiLSTM–CRF | 0.862±0.008 | 0.873±0.008 | 0.868±0.008 | 0.472±0.014 | 0.912±0.006 |
| Transformer-Random | 0.848±0.008 | 0.863±0.010 | 0.855±0.009 | 0.519±0.019 | 0.896±0.010 |
| Transformer-Char2Vec | 0.844±0.005 | 0.860±0.012 | 0.852±0.008 | 0.491±0.010 | 0.895±0.012 |
| TangutEncoder | 0.881±0.004 | 0.885±0.002 | 0.883±0.003 | 0.570±0.011 | 0.915±0.004 |
| CRF + Dict_all | 0.903±0.004 | 0.902±0.004 | 0.902±0.004 | 0.469±0.018 | 0.943±0.003 |
| CRF + Dict_all + Dist_all | 0.907±0.004 | 0.906±0.004 | 0.907±0.004 | 0.491±0.016 | 0.946±0.002 |
| TangutEncoder + Dict_all | 0.902±0.003 | 0.913±0.006 | 0.907±0.004 | 0.606±0.014 | 0.942±0.005 |
| TangutEncoder + Dict_all + Dist_all | **0.907±0.002** | **0.917±0.004** | **0.912±0.003** | **0.617±0.012** | 0.946±0.003 |

主要观察如下：

- 词典格网为Linear CRF带来最大的初始增益；
- 可靠度和辞书元数据有助于区分不同类型的候选词；
- 无标注语料统计进一步改善CRF；
- Char2Vec优于随机初始化，但明显弱于MLM上下文预训练；
- TangutEncoder的主要优势体现在有标注训练词表之外的词形召回；
- 显式词典知识在上下文预训练后仍然具有互补价值。
