"""Local Anima bundle verification and the installed Diffusers execution path."""
import hashlib
import json
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace

COMPONENTS = ('text_encoder','tokenizer','t5_tokenizer','vae','text_conditioner','transformer','scheduler')


@contextmanager
def progress_reporting(pipe,on_step):
    from diffusers.modular_pipelines.anima.denoise import AnimaDenoiseLoopWrapper
    # blocks returns a deepcopy; only _blocks is executed by the installed modular pipeline.
    pending=[pipe._blocks];seen=set()
    with ExitStack() as restore:
        while pending:
            block=pending.pop()
            if id(block) in seen:continue
            seen.add(id(block));pending.extend(block.sub_blocks.values())
            if not isinstance(block,AnimaDenoiseLoopWrapper):continue
            original=block.progress_bar
            @contextmanager
            def progress_bar(*args,_original=original,**kwargs):
                with _original(*args,**kwargs) as bar:
                    completed=0
                    on_step(completed,bar.total)
                    def update(count=1):
                        nonlocal completed
                        bar.update(count);completed+=count
                        on_step(completed,bar.total)
                    yield SimpleNamespace(update=update,total=bar.total)
            if 'progress_bar' in vars(block):restore.callback(setattr,block,'progress_bar',vars(block)['progress_bar'])
            else:restore.callback(delattr,block,'progress_bar')
            block.progress_bar=progress_bar
        yield


def verify_bundle(folder, check_hashes=True, cache=None):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'download_manifest.json').read_text(encoding='utf-8'))
    seen=set()
    for record in manifest['files']:
        relative=record['path'];path=(folder/relative).resolve()
        if not path.is_relative_to(folder) or relative in seen:
            raise ValueError('Invalid or duplicate bundle path')
        seen.add(relative)
        if not path.is_file():raise ValueError('Missing bundle file: '+relative)
        if check_hashes:
            if cache is not None:checksum=cache.file_hash(path)
            else:
                with path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
            if checksum!=record['sha256']:raise ValueError('Anima bundle hash mismatch: '+relative)
    required={'modular_model_index.json','scheduler/scheduler_config.json'}
    for component in ('text_encoder','vae','text_conditioner','transformer'):
        required.update((component+'/config.json',component+('/model.safetensors' if component=='text_encoder' else '/diffusion_pytorch_model.safetensors')))
    for component in ('tokenizer','t5_tokenizer'):
        required.update((component+'/tokenizer.json',component+'/tokenizer_config.json'))
    if not required <= seen:
        raise ValueError('Incomplete Anima bundle')
    index=json.loads((folder/'modular_model_index.json').read_text(encoding='utf-8'))
    if index.get('_class_name')!='AnimaModularPipeline':
        raise ValueError('Not an Anima pipeline bundle')
    return manifest


def split_single_state(state):
    transformer,conditioner={},{}
    for key,value in state.items():
        prefix=next((p for p in ('net.','model.diffusion_model.') if key.startswith(p)),None)
        if prefix is None:raise ValueError('Unknown Anima checkpoint key: '+key)
        key=key.removeprefix(prefix)
        destination=conditioner if key.startswith('llm_adapter.') else transformer
        key=key.removeprefix('llm_adapter.') if destination is conditioner else 'net.'+key
        if key in destination:raise ValueError('Duplicate Anima key: '+key)
        destination[key]=value
    if not transformer or not conditioner:raise ValueError('Missing Anima transformer/conditioner')
    return transformer,conditioner


def load_single_components(path,folder):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from diffusers import CosmosTransformer3DModel,AnimaTextConditioner
    from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers
    transformer_state,conditioner_state=split_single_state(load_file(str(path),device='cpu'))
    transformer_state=convert_cosmos_transformer_checkpoint_to_diffusers(transformer_state)
    result={}
    for name,cls,state in [('transformer',CosmosTransformer3DModel,transformer_state),('text_conditioner',AnimaTextConditioner,conditioner_state)]:
        with init_empty_weights():model=cls.from_config(json.loads((Path(folder)/name/'config.json').read_text(encoding='utf-8')))
        model.load_state_dict(state,strict=True,assign=True)
        result[name]=model.to(dtype=torch.bfloat16)
    return result


def load_pipeline(folder, guidance=4.0, checkpoint=None, loras=(), resolution=(1024,1024)):
    import torch
    from diffusers import AnimaModularPipeline, ComponentsManager, ClassifierFreeGuidance
    folder=Path(folder).resolve()
    manager=ComponentsManager()
    pipe=AnimaModularPipeline(components_manager=manager)
    for name in COMPONENTS:
        if checkpoint and name in ('transformer','text_conditioner'):continue
        # Use installed classes and local paths, never repository pointers from the index.
        spec=pipe.get_component_spec(name)
        spec.pretrained_model_name_or_path=str(folder)
        spec.subfolder=name
        spec.revision=None
        options={'local_files_only':True}
        if issubclass(spec.type_hint,torch.nn.Module):options['dtype']=torch.bfloat16
        pipe.update_components(**{name:spec.load(**options)})
    if checkpoint:pipe.update_components(**load_single_components(checkpoint,folder))
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=guidance))
    from .lora import apply_loras
    apply_loras(pipe,loras)
    pipe.vae.enable_tiling()
    pipe.anima_manager=manager
    configure_pipeline(pipe,resolution)
    return pipe, manager


def configure_pipeline(pipe,resolution):
    from .memory_policy import available_vram,available_ram,residency_plan
    free,total,measurement=available_vram()
    weights=sum(p.numel()*p.element_size() for component in ('text_encoder','vae','text_conditioner','transformer')
                for p in getattr(pipe,component).parameters())
    policy=residency_plan(free,total,weights,*resolution,free_ram=available_ram())
    policy.update(mode='auto_model_offload',measurement=measurement,family='anima')
    # The manager rechecks CUDA free memory per component; account for WDDM's driver discrepancy.
    margin=policy['activation_reserve_bytes']+max(0,measurement['cuda_free_bytes']-free)
    pipe.anima_manager.enable_auto_cpu_offload(device='cuda',memory_reserve_margin=margin)
    pipe.memory_policy=policy


def bundle_fingerprint(folder):
    folder=Path(folder)
    manifest=verify_bundle(folder,check_hashes=False)
    files={}
    for name in ['download_manifest.json']+[r['path'] for r in manifest['files']]:
        stat=(folder/name).stat()
        files[name]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    return {'files':files}


def count_prompt_tokens(folder,positive,negative,encoders=None):
    if encoders is None:
        from transformers import Qwen2Tokenizer,T5Tokenizer
        encoders=[Qwen2Tokenizer.from_pretrained(str(Path(folder)/'tokenizer'),local_files_only=True),
                  T5Tokenizer.from_pretrained(str(Path(folder)/'t5_tokenizer'),local_files_only=True)]
    rows=[]
    for name,tokenizer in zip(('Qwen3','T5 tokens'),encoders):
        counts={kind:len(tokenizer(text,truncation=False)['input_ids']) for kind,text in [('positive',positive),('negative',negative)]}
        rows.append(dict(encoder=name,limit=512,**counts,truncated=max(counts.values())>512))
    return dict(encoders=rows,truncated=any(row['truncated'] for row in rows))
