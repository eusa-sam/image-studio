"""Bounded weighted CLIP chunks, with independent tokenization for each encoder."""
import math
import re


def parse_weights(text):
    if '\0' in text:raise ValueError('NUL 문자는 허용하지 않습니다.')
    position=0

    def group(end=None,depth=0):
        nonlocal position
        if depth>16:raise ValueError('프롬프트 괄호는 최대16단계입니다.')
        parts=[]
        while position<len(text):
            char=text[position];position+=1
            if char=='\\':
                if position>=len(text):raise ValueError('프롬프트 끝의 escape가 불완전합니다.')
                parts.append(('\0' if text[position]==':' else text[position],1.0));position+=1
            elif char in '([':
                start=position
                nested=group(')' if char=='(' else ']',depth+1)
                factor=1.1 if char=='(' else 1/1.1
                if char=='(' and nested:
                    suffix=[]
                    for value,weight in reversed(nested):
                        if weight!=1:break
                        suffix.append(value)
                    tail=''.join(reversed(suffix))
                    match=re.search(r':([+-]?(?:\d+(?:\.\d*)?|\.\d+))$',tail)
                    if match and text[start:position-1].endswith(match[0]):
                        count=len(match[0]);factor=float(match[1])
                        while count:
                            value,weight=nested.pop()
                            if weight!=1:raise ValueError('가중치 표기 위치를 확인하세요.')
                            count-=len(value)
                for value,weight in nested:parts.append((value,weight*factor))
            elif char in ')]':
                if char!=end:raise ValueError('프롬프트 괄호가 맞지 않습니다.')
                return parts
            else:parts.append((char,1.0))
        if end:raise ValueError('닫히지 않은 프롬프트 괄호가 있습니다.')
        return parts

    merged=[]
    for value,weight in group():
        value=value.replace('\0',':')
        if not math.isfinite(weight) or not 0<=weight<=4:raise ValueError('프롬프트 유효 가중치는0~4입니다.')
        if merged and merged[-1][1]==weight:merged[-1]=(merged[-1][0]+value,weight)
        else:merged.append((value,weight))
    return merged


def weighted_tokens(tokenizer,text):
    tokens=[];weights=[]
    for value,weight in parse_weights(text):
        ids=tokenizer(value,add_special_tokens=False,truncation=False)['input_ids']
        tokens.extend(ids);weights.extend([weight]*len(ids))
    if len(tokens)>600:raise ValueError('긴 프롬프트 모드는 인코더별 최대600토큰입니다.')
    return tokens,weights


def encode_weighted(pipe,positive,negative,family):
    import torch
    encoders=[pipe.text_encoder]+([pipe.text_encoder_2] if family=='sdxl' else [])
    tokenizers=[pipe.tokenizer]+([pipe.tokenizer_2] if family=='sdxl' else [])
    prepared=[[weighted_tokens(t,text) for text in (positive,negative)] for t in tokenizers]
    chunks=max(1,max((len(ids)+74)//75 for pair in prepared for ids,_ in pair))
    all_embeddings=[[],[]];pooled=[None,None]
    with torch.inference_mode():
        for encoder_index,(encoder,tokenizer,pair) in enumerate(zip(encoders,tokenizers,prepared)):
            for side,(ids,weights) in enumerate(pair):
                embeddings=[]
                for chunk in range(chunks):
                    block=ids[chunk*75:(chunk+1)*75];w=weights[chunk*75:(chunk+1)*75]
                    pad=75-len(block)
                    tokens=[tokenizer.bos_token_id]+block+[tokenizer.eos_token_id]+[tokenizer.pad_token_id]*pad
                    tensor=torch.tensor([tokens],device=pipe._execution_device,dtype=torch.long)
                    kwargs={}
                    if getattr(encoder.config,'use_attention_mask',False):
                        kwargs['attention_mask']=torch.tensor([[1]*(len(block)+2)+[0]*pad],device=tensor.device)
                    output=encoder(tensor,output_hidden_states=family=='sdxl',**kwargs)
                    hidden=output.hidden_states[-2] if family=='sdxl' else output[0]
                    scale=torch.tensor([1]+w+[1]*(pad+1),device=hidden.device,dtype=hidden.dtype)
                    embeddings.append(hidden*scale[None,:,None])
                    if encoder_index==1 and chunk==max(0,(len(ids)-1)//75):pooled[side]=output.text_embeds
                all_embeddings[side].append(torch.cat(embeddings,dim=1))
    result=dict(prompt_embeds=torch.cat(all_embeddings[0],dim=-1),negative_prompt_embeds=torch.cat(all_embeddings[1],dim=-1))
    if family=='sdxl':result.update(pooled_prompt_embeds=pooled[0],negative_pooled_prompt_embeds=pooled[1])
    if any(not torch.isfinite(tensor).all() for tensor in result.values()):raise ValueError('Invalid prompt embeddings')
    return result
