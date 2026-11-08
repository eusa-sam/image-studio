"""One validated pipeline, owned by one serialized worker process."""
import json
import hashlib
import time


def file_stamp(path):
    import os
    if os.name!='nt':
        stat=path.stat()
        return stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev
    import ctypes as c
    import msvcrt
    from ctypes import wintypes as w
    class Basic(c.Structure):
        _fields_=[(name,c.c_longlong) for name in ('creation','access','write','change')]+[('attributes',w.DWORD)]
    class FileId(c.Structure):
        _fields_=[('volume',c.c_ulonglong),('identifier',c.c_ubyte*16)]
    api=c.WinDLL('kernel32',use_last_error=True)
    query=api.GetFileInformationByHandleEx
    query.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
    query.restype=w.BOOL
    try:
        with path.open('rb') as stream:
            info=Basic();identity=FileId()
            handle=w.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
            for kind,value in ((0,info),(18,identity)):
                if not query(handle,kind,c.byref(value),c.sizeof(value)):return None
            if info.change<=0 or not any(identity.identifier):return None
            # Timestamp ticks can collide during rapid edits; require the file's journal sequence too.
            control=api.DeviceIoControl
            control.argtypes=[w.HANDLE,w.DWORD,c.c_void_p,w.DWORD,c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.c_void_p]
            control.restype=w.BOOL
            versions=(c.c_ushort*2)(2,3);record=c.create_string_buffer(4096);returned=w.DWORD()
            if not control(handle,0x000900EB,versions,c.sizeof(versions),record,len(record),c.byref(returned),None):return None
            import struct
            if returned.value<8:return None
            length,major,minor=struct.unpack_from('<IHH',record)
            offset={2:24,3:40}.get(major)
            if offset is None or length>returned.value or length<offset+8:return None
            usn=struct.unpack_from('<q',record,offset)[0]
            size=os.fstat(stream.fileno()).st_size
            if usn<=0:
                # Journaling can be disabled. Small files are cheap to check in full;
                # recent large files cannot safely reuse a timestamp-only receipt.
                if size<=1024*1024:
                    usn=hashlib.sha256(stream.read()).hexdigest()
                elif not 0 <= info.change*100-11644473600000000000 < time.time_ns()-2_000_000_000:
                    return None
            return size,info.write,info.creation,bytes(identity.identifier).hex(),identity.volume,info.change,usn
    except OSError:
        # An unavailable Windows change stamp must not authorize a cached digest.
        return None


class PipelineCache:
    def __init__(self,receipt_root=None):
        self.receipt_root=receipt_root
        self.key=None
        self.pipe=None
        self.conditioning={}
        self.positive=None
        self.verified={}
        self.hash_checks=[]
        self.suspended=False
        self.lora_weights=None

    def file_hash(self,path):
        started=time.perf_counter()
        cpu_started=time.thread_time()
        path=path.resolve()
        from backend.app.verification_receipts import hash_reader
        with hash_reader(path) as stream:
            stat=path.stat()
            stamp=file_stamp(path)
            previous=self.verified.get(path)
            from backend.app.verification_receipts import read_receipt, write_receipt
            reused=stamp is not None and previous is not None and previous[0]==stamp
            source='memory' if reused else 'receipt'
            result=previous[1] if reused else read_receipt(self.receipt_root,path,stamp)
            reused=result is not None
            if not reused:
                source='read'
                result=hashlib.file_digest(stream,'sha256').hexdigest()
            after=path.stat()
            basic=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev)
            final=(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino,after.st_dev)
            ending=file_stamp(path)
            if (stamp is None and basic!=final) or (stamp is not None and ending!=stamp):
                raise ValueError('COMPONENT_CHANGED during hashing')
            if stamp is not None:self.verified[path]=(stamp,result)
            else:self.verified.pop(path,None)
            if source=='read':write_receipt(self.receipt_root,path,stamp,result)
            check=dict(path=str(path),bytes=stat.st_size,reused=reused,source=source,seconds=time.perf_counter()-started,
                       cpu_seconds=time.thread_time()-cpu_started,cpu_clock='thread')
            self.hash_checks.append(check)
            print(json.dumps(dict(event='file_hash',**check)),flush=True)
            return result

    @staticmethod
    def identity(plan,resolution):
        fields=('model_path','fingerprint','krea2','anima','bundle_fingerprint','config_sha256','components','loras','inpaint_channels','vision_config_sha256')
        values={key:plan.get(key) for key in fields}
        values['components']={key:value for key,value in (plan.get('components') or {}).items() if key!='upscaler'}
        if plan.get('runner','').startswith(('sdxl_','sd1_','sd2_')):
            values['prediction_type']=plan.get('settings',{}).get('prediction_type')
        if plan.get('runner','').startswith(('sdxl_','sd1_','sd2_','flux2_','zimage_','anima_','krea2_')):
            values['loras']=[{key:value for key,value in row.items() if key!='weight'} for row in (plan.get('loras') or [])]
        return json.dumps([values,resolution],sort_keys=True)

    def clear(self):
        try:
            if self.pipe is not None:
                from .execution import release
                release(self.pipe)
        finally:
            self.pipe=None
            self.key=None
            self.conditioning.clear()
            self.positive=None
            self.suspended=False
            self.lora_weights=None

    def suspend(self):
        if self.pipe is None:return False
        import gc,torch
        from .execution import enter_phase,finish_transfers
        finish_transfers(self.pipe)
        enter_phase(self.pipe,'suspend')
        if self.pipe.memory_policy.get('family') in ('flux2','zimage','sdxl','sd1','sd2','anima'):
            from backend.app.inference.runtime.memory_policy import available_ram
            from backend.app.inference.runtime.cpu_weights import backed_tensors, restore
            backed=backed_tensors(self.pipe)
            modules=[module for module in self.pipe.components.values() if isinstance(module,torch.nn.Module)]
            moving=sum(value.numel()*value.element_size() for module in modules for value in list(module.parameters())+list(module.buffers()) if value.is_cuda and id(value) not in backed)
            moving+=sum(value.numel()*value.element_size() for values in self.conditioning.values() for value in values.values() if value.is_cuda)
            if available_ram()<8*1024**3+moving:self.clear()
            else:
                if self.pipe.memory_policy.get('family')=='anima':self.pipe.anima_manager.disable_auto_cpu_offload()
                else:self.pipe.remove_all_hooks()
                restore(self.pipe)
                self.pipe.to('cpu')
                self.suspended=True
        elif self.pipe.memory_policy['mode']=='partial':
            for mapping in self.pipe.weight_maps:mapping.offload_cache()
            self.suspended=True
        else:self.clear()
        if self.suspended:
            self.conditioning={prompt:{name:value.cpu() for name,value in values.items()} for prompt,values in self.conditioning.items()}
        gc.collect();torch.cuda.empty_cache()
        return self.suspended

    def get(self,key,positive,*,replace_loras=None):
        # Klein/Z validation permits transformer LoRA only; encoded text is unchanged.
        encoder_lora=getattr(self.pipe,'memory_policy',{}).get('family') not in ('flux2','zimage','krea2')
        if self.pipe is not None and self.key is not None and key!=self.key and getattr(self.pipe,'memory_policy',{}).get('family') in ('sd1','sd2','sdxl','flux2','zimage','anima'):
            previous=json.loads(self.key);requested=json.loads(key)
            if previous[0]==requested[0] and previous[1]!=requested[1]:
                # Keep weights, but force the worker to re-plan residency for the new image size.
                if not self.suspended and not self.suspend():return None
                self.key=key
        if replace_loras is not None and self.pipe is not None and self.key is not None and key!=self.key:
            previous=json.loads(self.key);requested=json.loads(key)
            adapters=previous[0].get('loras') or []
            previous[0]['loras']=[]
            requested[0]['loras']=[]
            if previous==requested:
                try:
                    if getattr(self.pipe,'memory_policy',{}).get('family')=='krea2' and self.pipe.memory_policy['mode']=='partial':
                        from backend.app.inference.models.krea2_lora import replace_loras as replace_krea_loras
                        replace_krea_loras(self.pipe,replace_loras)
                        self.key=key
                        self.lora_weights=tuple(selection['weight'] for selection,_ in replace_loras)
                        return self.get(key,positive)
                    # Re-plan new adapters from CPU; ANIMA also needs its manager disabled for removal.
                    replan=replace_loras or getattr(self.pipe,'memory_policy',{}).get('family')=='anima' or getattr(self.pipe,'_cpu_weights',None)
                    if replan and not self.suspended and not self.suspend():return None
                    from backend.app.inference.runtime.cpu_weights import forget
                    forget(self.pipe)
                    if adapters:self.pipe.delete_adapters([f'lora_{i}' for i in range(len(adapters))])
                    if replace_loras:
                        from backend.app.inference.conditioning.lora import apply_loras
                        apply_loras(self.pipe,replace_loras)
                    if encoder_lora:self.conditioning.clear()
                except Exception:
                    self.clear()
                    raise
                self.key=key
        if key!=self.key:self.clear()
        if replace_loras is not None:
            weights=tuple(selection['weight'] for selection,_ in replace_loras)
            if weights!=self.lora_weights:
                if encoder_lora:self.conditioning.clear()
                if self.pipe is not None:
                    from .execution import finish_transfers
                    finish_transfers(self.pipe)
                    for mapping in getattr(self.pipe,'weight_maps',()):
                        if hasattr(mapping,'derived_identity'):
                            from backend.app.inference.quantization.lora_weights import invalidate
                            invalidate(mapping)
                    # Reattach CPU originals before invalidation would force a new device-to-host copy.
                    if getattr(self.pipe,'_cpu_weights',None) and not self.suspended and not self.suspend():
                        self.lora_weights=weights
                        return None
                    from backend.app.inference.runtime.cpu_weights import forget
                    forget(self.pipe)
                self.lora_weights=weights
        if positive!=self.positive:
            self.conditioning.clear()
            self.positive=positive
        return self.pipe

