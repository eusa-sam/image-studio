"""Explicit installation-time export of pinned face/hand detectors; never called by inference."""
import os
from pathlib import Path
import json
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]
REVISION = '53cc19de382014514d9d4038601d261a7faa9b7b'
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'data/runtime/ultralytics')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    if __package__:
        from .downloads import hf_hub_download
    else:
        from downloads import hf_hub_download
    import ultralytics
    import torch
    import onnxruntime
    import numpy as np
    source_root = ROOT / 'models/sources/adetailer' / REVISION
    target_root = ROOT / 'models/detectors'
    source_root.mkdir(parents=True,exist_ok=True)
    target_root.mkdir(parents=True,exist_ok=True)
    for target in ('face','hand'):
        name = target + '_yolov8n.pt'
        downloaded = hf_hub_download('Bingsu/adetailer',name,revision=REVISION,token=False)
        original = source_root / name
        if not original.exists():
            shutil.copyfile(downloaded,original)
        elif checksum(original) != checksum(Path(downloaded)):
            raise ValueError('Existing source differs: '+str(original))
        exported = original.with_suffix('.onnx')
        if not exported.exists():
            model = ultralytics.YOLO(str(original))
            model.export(format='onnx',imgsz=640,batch=1,dynamic=False,simplify=False,opset=17,device='cpu',nms=False)
        session = onnxruntime.InferenceSession(str(exported),providers=['CPUExecutionProvider'])
        result = session.run(None,{session.get_inputs()[0].name:np.zeros((1,3,640,640),dtype=np.float32)})
        if len(result) != 1 or result[0].shape != (1,5,8400) or not np.isfinite(result[0]).all():
            raise ValueError('Unexpected detector layout')
        destination = target_root / exported.name
        if destination.exists() and checksum(destination) != checksum(exported):
            raise ValueError('Refusing to overwrite detector')
        if not destination.exists():
            shutil.copyfile(exported,destination)
        profile = dict(schema_version=1,kind='detector',family='yolov8',scale=1,target=target,
                       sha256=checksum(destination),source=f'https://huggingface.co/Bingsu/adetailer/blob/{REVISION}/{name}')
        profile_path = destination.with_name(destination.name+'.profile.json')
        if not profile_path.exists():
            profile_path.write_text(json.dumps(profile,indent=2),encoding='utf-8')
        manifest = dict(source_sha256=checksum(original),onnx_sha256=checksum(destination),revision=REVISION,
                        ultralytics=ultralytics.__version__,torch=torch.__version__,opset=17,input=[1,3,640,640],output=[1,5,8400])
        manifest_path = source_root / (target+'.export.json')
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(destination,profile['sha256'],flush=True)


if __name__ == '__main__':
    main()
