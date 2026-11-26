# Qwen through XMSXB

The platform documents OpenAI-compatible POST /v1/chat/completions:
https://docs.xmsxb.com/api/openai/chat.html
Model catalog: https://docs.xmsxb.com/guide/models.html

Configure the actual secret associated with your named key (the display name is
not the secret). No credentials are stored in the repository:

```bash
export XMSXB_API_KEY='<actual API key>'
export XMSXB_BASE_URL='https://zyapi.xmsxb.com/v1'
python -m review --data /path/to/video.mp4 --out /path/to/job \
  --rig ego_head --no-board --model qwen3.6-plus --reasoning none \
  --sample-fps 2 --max-tokens 16384 --concurrency 1 --cap 0
```

L1/L3 adapters accept the same model options. Existing completed jobs can be
restarted with --force. Both video annotation and structured extraction use the
same gateway and model. XMSXB_BASE_URL accepts a root URL, /v1, or full completion
endpoint. The default is https://zyapi.xmsxb.com/v1.

The qwen3.6-plus route requires XMSXB_API_KEY and never falls back to old Alibaba
credentials. Other legacy Qwen deployments retain their existing configuration.
DashScope-specific chat_template_kwargs are omitted for gateway requests.
Old qwen3.8-27b list-price estimates do not apply; do not use --estimated-cap-cny.
--cap 0 leaves cost control to the provider account; no replacement prices are assumed.

The multimodal request retains sampled image_url blocks. Actual image support,
model access, and output limits require a live gateway test with your key; local
mock tests do not certify them. Claude Code/OpenClaw installation is unnecessary.
