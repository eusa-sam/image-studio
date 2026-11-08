"""Validated explicit-token embeddings for SD encoders; no implicit prompt rewriting."""
import copy
import hashlib
import unicodedata


def read_embedding(path, profile):
    import torch
    from safetensors import SafetensorError
    from safetensors.torch import load
    family=profile.get('family')
    widths={'sd1':{'emb_params':768},'sd2':{'emb_params':1024},'sdxl':{'clip_l':768,'clip_g':1280}}.get(family)
    tokens=profile.get('tokens')
    if widths is None:raise ValueError('임베딩은 SD1/SD2/SDXL만 지원합니다.')
    if (not isinstance(tokens,list) or not 1<=len(tokens)<=16 or
        any(not isinstance(t,str) or not 1<=len(t)<=128 or any(c.isspace() or unicodedata.category(c).startswith('C') for c in t) for t in tokens) or
        len(set(tokens))!=len(tokens)):
        raise ValueError('임베딩 토큰은 중복 없는 1~16개, 각각 공백 없는 1~128자여야 합니다.')
    with path.open('rb') as stream:content=stream.read(4*1024**2+1)
    if len(content)>4*1024**2:raise ValueError('임베딩 파일은 4MiB 이하여야 합니다.')
    if hashlib.sha256(content).hexdigest()!=profile.get('sha256'):raise ValueError('EMBEDDING_HASH_MISMATCH')
    try:weights=load(content)
    except SafetensorError as error:raise ValueError('잘못된 임베딩 safetensors 파일입니다.') from error
    if set(weights)!=set(widths):raise ValueError('임베딩 encoder 키가 계열과 맞지 않습니다.')
    for key,width in widths.items():
        value=weights[key]
        if tuple(value.shape)!=(len(tokens),width) or value.dtype not in (torch.float16,torch.bfloat16,torch.float32) or not torch.isfinite(value).all():
            raise ValueError('임베딩 shape/dtype/유한값을 확인하세요: '+key)
    return weights


def validated_embeddings(selections, family, tokenizers):
    if len(selections)>4:raise ValueError('임베딩은 최대 4개까지 선택할 수 있습니다.')
    vocab=set().union(*(tokenizer.get_vocab() for tokenizer in tokenizers))
    result=[]
    for component,path in selections:
        profile=component['profile']
        if profile['family']!=family:raise ValueError('임베딩 계열이 현재 모델과 다릅니다.')
        weights=read_embedding(path,profile)
        tokens=profile['tokens']
        if any(token in vocab for token in tokens):raise ValueError('임베딩 토큰이 기존 단어 또는 다른 임베딩과 충돌합니다.')
        vocab.update(tokens);result.append((tokens,weights))
    return result


def embedding_tokenizers(family, selections):
    from backend.app.inference.conditioning.prompt_tokens import tokenizers
    original=tokenizers(family)
    prepared=validated_embeddings(selections,family,original)
    result=copy.deepcopy(original)
    for tokens,_ in prepared:
        for tokenizer in result:tokenizer.add_tokens(tokens)
    return result


def apply_embeddings(pipe, selections, family):
    import torch
    slots=[('clip_l' if family=='sdxl' else 'emb_params',pipe.tokenizer,pipe.text_encoder)]
    if family=='sdxl':slots.append(('clip_g',pipe.tokenizer_2,pipe.text_encoder_2))
    prepared=validated_embeddings(selections,family,[slot[1] for slot in slots])
    # Validate every encoder and conversion before mutating either vocabulary.
    for _,weights in prepared:
        for key,_,encoder in slots:
            target=encoder.get_input_embeddings().weight
            if target.shape[1]!=weights[key].shape[1] or not torch.isfinite(weights[key].to(dtype=target.dtype)).all():
                raise ValueError('선택 CLIP의 임베딩 차원/정밀도와 호환되지 않습니다.')
    for tokens,weights in prepared:
        for key,tokenizer,encoder in slots:
            pipe.load_textual_inversion(weights[key],token=tokens,text_encoder=encoder,tokenizer=tokenizer)
