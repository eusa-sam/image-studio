"""Small real CLIP tokenizer files, independent of downloaded model assets."""
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from backend.app.inference.conditioning import prompt_tokens


@contextmanager
def local_tokenizers():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Two merges make 'cat' one token, allowing exact 75/76-token boundaries.
        vocab = {word: index for index, word in enumerate(
            ['<|startoftext|>', '<|endoftext|>', 'c', 'a', 't</w>', 'ca', 'cat</w>'])}
        for family, names in [('sd1', ['tokenizer']), ('sdxl', ['tokenizer', 'tokenizer_2'])]:
            for name in names:
                folder = root / 'configs' / family / name
                folder.mkdir(parents=True)
                (folder / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
                (folder / 'merges.txt').write_text('#version: 0.2\nc a\nca t</w>\n', encoding='utf-8')
                (folder / 'tokenizer_config.json').write_text(
                    json.dumps({'model_max_length': 77}), encoding='utf-8')
        prompt_tokens.tokenizers.cache_clear()
        try:
            with patch.object(prompt_tokens, 'MODEL_ROOT', root):
                yield root
        finally:
            prompt_tokens.tokenizers.cache_clear()
