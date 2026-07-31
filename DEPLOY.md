# 西夏文分词服务部署指南

## 1. 环境准备

```bash
git clone <仓库地址>
cd xixia_seg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 获取模型文件

模型文件较大（~200MB），不放在 git 中。从以下渠道获取 `saved_models/` 目录：

- **局域网共享**: `/disks/sdb/user_space/denglifan/workspace/xixia_seg/saved_models/`
- **scp 拷贝**: `scp -r user@host:/path/to/saved_models/ ./saved_models/`

确保包含以下三个文件：
```
saved_models/
├── TEnc-MLM+dict+gap_model.pt
├── TEnc-MLM+dict+gap_lexicon.pkl
└── TEnc-MLM+dict+gap_gap.pkl
```

## 3. 启动服务

```bash
python serve.py --model TEnc-MLM+dict+gap --port 8000
```

## 4. 调用示例

```bash
# 单句分词
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{"text": "𘝵𗯩𗰭𗏣𗅋𘙰𗢳𗂧𗅁𗥩𗄭"}'

# 批量分词
curl -X POST http://localhost:8000/segment/batch \
  -H "Content-Type: application/json" \
  -d '{"sentences": ["𘝵𗯩", "𗏣𗅋"]}'

# Python 调用
import requests
resp = requests.post("http://localhost:8000/segment",
    json={"text": "𘝵𗯩𗰭𗏣𗅋𘙰𗢳𗂧𗅁𗥩𗄭"})
print(resp.json()["segmented"])
```
