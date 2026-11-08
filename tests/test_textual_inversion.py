import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import CLIPTextConfig,CLIPTextModel,PreTrainedTokenizerFast
from diffusers.loaders import TextualInversionLoaderMixin

from backend.app.inference.conditioning.textual_inversion import read_embedding, apply_embeddings, embedding_tokenizers


def tokenizer():
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]':0,'base':1},unk_token='[UNK]')),unk_token='[UNK]')


class Pipe(TextualInversionLoaderMixin):
    hf_device_map=None
    def __init__(self):
        self.tokenizer=tokenizer();self.tokenizer_2=tokenizer()
        self.text_encoder=CLIPTextModel(CLIPTextConfig(vocab_size=2,hidden_size=768,num_hidden_layers=0,num_attention_heads=1,max_position_embeddings=4,bos_token_id=0,eos_token_id=1))
        self.text_encoder_2=CLIPTextModel(CLIPTextConfig(vocab_size=2,hidden_size=1280,num_hidden_layers=0,num_attention_heads=1,max_position_embeddings=4,bos_token_id=0,eos_token_id=1))
        self.components={'text_encoder':self.text_encoder,'text_encoder_2':self.text_encoder_2}


class EmbeddingTests(unittest.TestCase):
    def write(self,folder,weights,tokens=None):
        path=folder/'embedding.safetensors';save_file(weights,str(path))
        return {'profile':{'family':'sdxl','tokens':tokens or ['<s0>','<s1>'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}},path

    def test_real_dual_encoder_injection_and_tokenizer_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            values={'clip_l':torch.ones(2,768),'clip_g':torch.full((2,1280),2.)}
            selection=self.write(Path(temp),values);pipe=Pipe()
            originals=[t.get_vocab().copy() for t in (pipe.tokenizer,pipe.tokenizer_2)]
            with patch('backend.app.inference.conditioning.prompt_tokens.tokenizers',return_value=[pipe.tokenizer,pipe.tokenizer_2]):
                clones=embedding_tokenizers('sdxl',[selection])
            self.assertEqual([t.get_vocab() for t in (pipe.tokenizer,pipe.tokenizer_2)],originals)
            self.assertTrue(all('<s0>' in t.get_vocab() for t in clones))
            before=[e.get_input_embeddings().weight.detach().clone() for e in (pipe.text_encoder,pipe.text_encoder_2)]
            apply_embeddings(pipe,[selection],'sdxl')
            for index,(key,t,e) in enumerate([('clip_l',pipe.tokenizer,pipe.text_encoder),('clip_g',pipe.tokenizer_2,pipe.text_encoder_2)]):
                weight=e.get_input_embeddings().weight
                self.assertTrue(torch.equal(weight[:2],before[index]))
                self.assertTrue(torch.equal(weight[[t.convert_tokens_to_ids(x) for x in ['<s0>','<s1>']]],values[key]))
            with self.assertRaisesRegex(ValueError,'충돌'):apply_embeddings(pipe,[selection],'sdxl')

    def test_invalid_later_selection_does_not_mutate_first_encoder(self):
        with tempfile.TemporaryDirectory() as temp:
            selection=self.write(Path(temp),{'clip_l':torch.ones(2,768),'clip_g':torch.ones(2,1280)})
            pipe=Pipe();before=pipe.tokenizer.get_vocab().copy()
            with self.assertRaisesRegex(ValueError,'충돌'):apply_embeddings(pipe,[selection,selection],'sdxl')
            self.assertEqual(pipe.tokenizer.get_vocab(),before)
            self.assertEqual(pipe.text_encoder.get_input_embeddings().weight.shape[0],2)

    def test_file_hash_shapes_tokens_and_values_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);good={'clip_l':torch.ones(2,768),'clip_g':torch.ones(2,1280)}
            for weights in ({'clip_l':good['clip_l']},dict(good,clip_g=torch.ones(1,1280)),dict(good,clip_l=torch.full((2,768),float('nan'))),dict(good,clip_l=torch.ones(2,768,dtype=torch.int32))):
                component,path=self.write(root,weights)
                with self.assertRaises(ValueError):read_embedding(path,component['profile'])
            component,path=self.write(root,good)
            for tokens in [['x','x'],['has space','x'],['\u200b','x']]:
                with self.assertRaises(ValueError):read_embedding(path,dict(component['profile'],tokens=tokens))
            with self.assertRaisesRegex(ValueError,'HASH'):read_embedding(path,dict(component['profile'],sha256='0'*64))
            pipe=Pipe()
            component,path=self.write(root,dict(good,clip_g=torch.full((2,1280),1e30)))
            pipe.text_encoder_2.half()
            with self.assertRaisesRegex(ValueError,'정밀도'):apply_embeddings(pipe,[(component,path)],'sdxl')
            self.assertNotIn('<s0>',pipe.tokenizer.get_vocab())
