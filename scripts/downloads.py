"""Project-owned preparation caches; never use the user's system-drive cache."""
import os
from pathlib import Path

CACHE_ROOT = Path(__file__).resolve().parents[1] / 'data/cache/huggingface'
# Set these before the first Xet session is created in a preparation process.
os.environ['HF_HOME'] = str(CACHE_ROOT)
os.environ['HF_HUB_CACHE'] = str(CACHE_ROOT / 'hub')
os.environ['HF_XET_CACHE'] = str(CACHE_ROOT / 'xet')


def hf_hub_download(*args, **kwargs):
    from huggingface_hub import hf_hub_download as download
    kwargs['cache_dir'] = str(CACHE_ROOT / 'hub')
    return download(*args, **kwargs)
