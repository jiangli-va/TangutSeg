"""tag_model - 联合分词+词性标注模型。

提供使用 BI+POS 大标签集的模型，一次性完成分词和词性标注。

模块:
    tag_utils.py         - POS 标签规范化、联合标签映射
    crf_joint.py         - CRF 联合分词+词性标注
    bilstm_crf_joint.py  - BiLSTM-CRF 联合分词+词性标注
"""

from tag_model.tag_utils import (
    normalize_tag, normalize_tags,
    build_joint_label_map,
    words_tags_to_bies_pos, bies_pos_to_words_tags,
)
from tag_model.crf_joint import CRFJointTagger
from tag_model.bilstm_crf_joint import BiLSTMCRFJointTagger
