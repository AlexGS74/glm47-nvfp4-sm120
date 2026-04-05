"""Quantize zai-org/GLM-4.7 (BF16) to FP8 block format using llmcompressor.

Preserves MTP (NextN) head at layer 92 in full precision for speculative
decoding compatibility with SGLang and vLLM.

Usage:
    python scripts/quant_glm47_fp8_block.py
    MODEL_IN=/path/to/GLM-4.7 MODEL_OUT=/path/to/GLM-4.7-FP8 python scripts/quant_glm47_fp8_block.py
"""
import os

from llmcompressor import model_free_ptq

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")

MODEL_IN = os.environ.get(
    "MODEL_IN",
    "/data/huggingface/hub/models--zai-org--GLM-4.7/snapshots/602d01efcdd332c5238ca4bcede555defbe83eb7",
)
MODEL_OUT = os.environ.get("MODEL_OUT", "/mnt/truenas-alexgs/hyperipper/models/GLM-4.7-FP8-block")

model_free_ptq(
    model_stub=MODEL_IN,
    save_directory=MODEL_OUT,
    scheme="FP8_BLOCK",
    ignore=[
        # Embeddings and output head
        "lm_head",
        "model.embed_tokens",
        # MoE router gates (small, precision-sensitive)
        "re:.*mlp\\.gate$",
        # MTP NextN head (layer 92) -- keep full precision for speculative decoding
        # Includes: enorm, hnorm, embed_tokens, shared_head.head, shared_head.norm,
        #           and the full transformer block (self_attn, mlp, norms)
        "re:model\\.layers\\.92\\.",
    ],
    max_workers=16,
    device="cuda:0",
)

print(f"SUCCESS: FP8 block quant saved to {MODEL_OUT}")

# Optional: upload to HuggingFace
HF_REPO = os.environ.get("HF_REPO", "AlexGS74/GLM-4.7-FP8-block")
answer = input(f"\nUpload to https://huggingface.co/{HF_REPO}? [y/N] ").strip().lower()
if answer == "y":
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO, exist_ok=True)
    api.upload_folder(folder_path=MODEL_OUT, repo_id=HF_REPO)
    print(f"Uploaded to https://huggingface.co/{HF_REPO}")
else:
    print(f"Skipped upload. To upload later:\n  hf upload {HF_REPO} {MODEL_OUT}")
