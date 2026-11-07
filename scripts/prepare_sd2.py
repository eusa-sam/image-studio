"""Pinned SD2.1 community mirror; original Stability AI repository is deprecated."""
from scripts.prepare_sd1 import main

if __name__ == '__main__':
    main(repo='sd2-community/stable-diffusion-2-1',revision='bb2154823665391b4fb29b0b9cf82a198964ee05',
         family='sd2',checkpoint='v2-1_768-ema-pruned.safetensors')
