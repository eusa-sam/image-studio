import unittest
from types import SimpleNamespace
import torch
from backend.app.inference.conditioning.prompt_encoding import parse_weights, weighted_tokens, encode_weighted


class Tokenizer:
    bos_token_id=1;eos_token_id=2;pad_token_id=0
    def __init__(self,repeat=1):self.repeat=repeat
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text for _ in range(self.repeat)]}


class Encoder:
    config=SimpleNamespace(use_attention_mask=False)
    def __call__(self,ids,output_hidden_states=False,**kwargs):
        value=ids.float().unsqueeze(-1)
        if output_hidden_states:return SimpleNamespace(hidden_states=[value,value+1,value+2],text_embeds=ids.float().sum(dim=1,keepdim=True))
        return (value,)


class EncodingTests(unittest.TestCase):
    def test_weights_nesting_escape_and_invalid_syntax(self):
        self.assertEqual(parse_weights('(red:1.5)'),[('red',1.5)])
        self.assertAlmostEqual(parse_weights('([red])')[0][1],1)
        self.assertEqual(parse_weights(r'\(red\)'),[('(red)',1)])
        self.assertEqual(parse_weights(r'(red\:2)'),[('red:2',1.1)])
        self.assertEqual(parse_weights('(red:2(blue))')[0],('red:2',1.1))
        self.assertEqual(parse_weights('(foo:(2:1))'),[('foo:2',1.1)])
        for text in ['(red','red)', '(red:5)','red'+chr(92)]:
            with self.assertRaises(ValueError):parse_weights(text)
        with self.assertRaisesRegex(ValueError,'600'):weighted_tokens(Tokenizer(),'x'*601)

    def test_sd1_keeps_tail_and_weights_only_selected_tokens(self):
        p=SimpleNamespace(text_encoder=Encoder(),tokenizer=Tokenizer(),_execution_device='cpu')
        encoded=encode_weighted(p,'a'*75+'(z:2)','',family='sd1')
        self.assertEqual(encoded['prompt_embeds'].shape,(1,154,1))
        self.assertEqual(encoded['negative_prompt_embeds'].shape,(1,154,1))
        self.assertEqual(encoded['prompt_embeds'][0,78,0],ord('z')*2)
        self.assertEqual(encoded['prompt_embeds'][0,1,0],ord('a'))

    def test_sdxl_independent_token_lengths_and_last_real_chunk_pooling(self):
        p=SimpleNamespace(text_encoder=Encoder(),text_encoder_2=Encoder(),tokenizer=Tokenizer(),tokenizer_2=Tokenizer(2),_execution_device='cpu')
        result=encode_weighted(p,'x'*76,'z',family='sdxl')
        self.assertEqual(result['prompt_embeds'].shape,(1,231,2))
        self.assertEqual(result['negative_prompt_embeds'].shape,(1,231,2))
        self.assertEqual(result['pooled_prompt_embeds'].item(),1+2*ord('x')+2)
        self.assertEqual(result['negative_pooled_prompt_embeds'].item(),1+2*ord('z')+2)
