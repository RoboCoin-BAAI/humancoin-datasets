# RoboCOIN-with-Depth（精简：文件系统 DA3 深度转化）

仅保留 FS 级深度转化相关代码（`da3` strategy），不含 RIFE / VDA / lingbot / LeRobot 写入路径。

## 依赖

- Conda 环境：本地示例为 `robocoin-depth`（`/opt/miniconda3/envs/robocoin-depth`）
- 共享目录环境：`/share/project/liuyou/humancoin/envs/`（当前有 `ego`）
- 需能 `import depth_anything_3`（把 `third_party/Depth-Anything-3/src` 加到 `PYTHONPATH`，或 `pip install -e third_party/Depth-Anything-3`）
- 模型权重：`DA3NESTED-GIANT-LARGE-1.1`（本地或 HF）

```bash
export PYTHONPATH="$(pwd)/third_party/Depth-Anything-3/src:${PYTHONPATH:-}"
```

## 运行

```bash
# 单 session，深度写到独立目录，只转 head/eye
python scripts/add_depth_fs_parallel.py \
  --dataset-dir /path/to/session \
  --output-dir  /path/to/session_depth \
  --cameras head,eye \
  --strategy da3 \
  --da3-model depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --da3-process-res 504 --da3-chunk-size 420 --da3-overlap 140 \
  --crf 6 --preset slow

# 批量
CAMERAS=head,eye bash scripts/run_sessions_depth_batch.sh \
  /path/to/session-episode \
  /path/to/session-episode_depth
```
