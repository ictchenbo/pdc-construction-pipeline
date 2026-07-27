# 按领域采样 20%（6～80 词过滤）
python3 src/04_annotation/sample.py --sample-fraction 0.2 --manifest data/validation/sample_manifest_w6-80.0.2_v1.jsonl --min-words 6

# 基于 gpt thinking(xhigh)进行标注
python3 src/04_annotation/annotate.py --provider gpt5 --manifest data/validation/sample_manifest_w6-80.0.2_v1.jsonl --thinking
