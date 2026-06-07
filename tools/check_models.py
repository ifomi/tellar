"""Check which candidate MLX models exist in the HF Hub and how big they are.

Metadata only — does NOT download any weights. Run from the project venv:

    cd ~/tellar && .venv/bin/python tools/check_models.py

Edit CANDIDATES to add/remove repos. Used to build the Studio-LLM A/B queue
(see plans/studio-llm.md §2) without guessing repo ids or sizes.
"""
from huggingface_hub import HfApi

# Candidate models for the Polish A/B. Gemma 4B first (Qwen dropped — the two
# we tried were unsatisfactory). 8B tier is a fallback; nothing 14B+ (16 GB RAM).
CANDIDATES = [
    # --- Gemma 4B (primary) ---
    "mlx-community/gemma-3-4b-it-qat-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-3-4b-it-bf16",
    # --- other small/mid multilingual (fallback tier) ---
    "mlx-community/gemma-2-9b-it-4bit",
    "mlx-community/aya-expanse-8b-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Ministral-8B-Instruct-2410-4bit",
]


def main():
    api = HfApi()
    for repo in CANDIDATES:
        try:
            info = api.model_info(repo, files_metadata=True)
            size_gb = sum((s.size or 0) for s in info.siblings) / 1e9
            print(f"OK    {repo:46}  ~{size_gb:4.1f} GB")
        except Exception as e:
            print(f"MISS  {repo:46}  {type(e).__name__}: {str(e)[:50]}")


if __name__ == "__main__":
    main()
