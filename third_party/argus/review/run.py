'''
PYTHONPATH=/share/project/liuyou/humancoin/code/argus \
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/argus/review/__main__.py \
  --data /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001/cp_0001_session_20261009_172052/videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4 \
  --rig ego_head \
  --no-board \
  --free


# Set DASHSCOPE_API_KEY in the terminal before running; do not store it here.
export DASHSCOPE_BASE_URL="https://llm-g3o8d3j71xbf6prc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

PYTHONPATH=/share/project/liuyou/humancoin/code/argus \
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/argus/review/__main__.py \
  --data /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001/cp_0001_session_20261009_172052/videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4 \
  --rig ego_head \
  --no-board \
  --model qwen3.8-27b \
  --reasoning none \
  --sample-fps 2 \
  --max-tokens 16384 \
  --concurrency 1 \
  --cap 0 \
  --estimated-cap-cny 0.6


# Batch: each session's rectified left-eye video is reviewed in its own process.
PYTHONPATH=/share/project/liuyou/humancoin/code/argus \
/share/project/liuyou/humancoin/envs/ego/bin/python \
  /share/project/liuyou/humancoin/code/argus/review/__main__.py \
  --source /share/project/liuyou/humancoin/data/L1_lerobot/batch_0000001 \
  --workers 8 \
  --rig ego_head \
  --no-board \
  --model qwen3.8-27b \
  --reasoning none \
  --sample-fps 2 \
  --max-tokens 16384 \
  --concurrency 1 \
  --cap 0 \
  --estimated-cap-cny 0.6

'''
