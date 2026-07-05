#!/usr/bin/env bash
# 环境补丁：解决三件在新机器上经常踩的事
#
# 1) huggingface.co 在国内/部分网络环境直连超时 → 用 hf-mirror 镜像
# 2) 系统 libstdc++ 太老（< 3.4.29），numpy 2.x 加载会爆"don't import from source dir" → 用 conda 自带的新版覆盖
# 3) FlagEmbedding 1.4 与 transformers 5.x 不兼容 → 在 requirements.txt 里锁了 <5.0，装依赖即可
#
# 用法：source env.sh     （不要用 ./ 执行，那样不会传到当前 shell）

# 1) HuggingFace 镜像（如果 hf.co 可达，注释掉这两行即可）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60

# 2) libstdc++ 覆盖 —— 路径按你机器上 conda 安装位置调整。
#    找新版：find ~/anaconda3 ~/miniconda3 /opt -name "libstdc++.so.6.0.3*" 2>/dev/null
if [ -z "$LD_PRELOAD" ]; then
    for cand in \
        /home/bibdr/anaconda3/lib/libstdc++.so.6 \
        "$HOME/anaconda3/lib/libstdc++.so.6" \
        "$HOME/miniconda3/lib/libstdc++.so.6" \
        /opt/conda/lib/libstdc++.so.6; do
        if [ -f "$cand" ]; then
            export LD_PRELOAD="$cand"
            break
        fi
    done
fi

# 3) 避免 tokenizers 跑并行时 fork 出多个进度条
export TOKENIZERS_PARALLELISM=false

echo "[env.sh] HF_ENDPOINT=$HF_ENDPOINT"
echo "[env.sh] LD_PRELOAD=$LD_PRELOAD"
