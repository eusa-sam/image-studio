"""Local SD1/SDXL token counts; no encoder weights or network requests."""
from functools import lru_cache
from backend.app.models import MODEL_ROOT


@lru_cache(maxsize=3)
def tokenizers(family='sdxl'):
    from transformers import CLIPTokenizer
    return [CLIPTokenizer.from_pretrained(str(MODEL_ROOT / 'configs' / family / name), local_files_only=True)
            for name in (('tokenizer', 'tokenizer_2') if family == 'sdxl' else ('tokenizer',))]


def count_tokens(positive, negative, encoders=None, family='sdxl',mode='native'):
    rows = []
    for name, tokenizer in zip(('OpenCLIP-H',) if family=='sd2' else ('CLIP-L', 'OpenCLIP bigG'), encoders if encoders is not None else tokenizers(family)):
        limit = tokenizer.model_max_length - 2
        if mode=='weighted':
            from backend.app.inference.conditioning.prompt_encoding import weighted_tokens
            counts={kind:len(weighted_tokens(tokenizer,text)[0]) for kind,text in (('positive',positive),('negative',negative))}
        else:
            counts = {kind:len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])
                      for kind,text in (('positive',positive),('negative',negative))}
        rows.append(dict(encoder=name, limit=limit, **counts,
                         chunks=max(1,max((count+limit-1)//limit for count in counts.values())),truncated=mode=='native' and any(count > limit for count in counts.values())))
    return {'encoders':rows, 'truncated':any(row['truncated'] for row in rows),'mode':mode}
