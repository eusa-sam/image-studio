"""Local single-class YOLOv8 detection and bounded region compositing."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def detail_prompt(target, positive, family):
    prefixes={'face':'detailed face, natural eyes, ','hand':'natural hand, anatomically correct fingers, '}
    if target not in prefixes:raise ValueError('Unsupported detail target')
    if family=='krea2' and target=='hand':
        # A whole-scene prompt can turn the enlarged hand crop into another portrait.
        return ('Close-up detail of the single hand already visible in the input crop. '
                'Preserve its exact position, wrist, gesture, finger directions, skin tone and lighting. '
                'Refine the existing fingers, natural joints and fingernails. '
                'Keep the surrounding clothing and background unchanged. '
                'This crop depicts a hand, not a portrait or a full person.')
    return prefixes[target]+positive


def select_regions(regions, limit, seed):
    return [dict(region,seed=(seed+index+1) % 2**32)
            for index,region in enumerate(sorted(regions,key=lambda r:r['confidence'],reverse=True)[:limit])]


def suppress(boxes, threshold=0.45):
    result = []
    for box in sorted(boxes,key=lambda b:b['confidence'],reverse=True):
        x1,y1,x2,y2 = box['box']
        area = (x2-x1)*(y2-y1)
        for prior in result:
            a,b,c,d = prior['box']
            intersection = max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
            if intersection / (area+(c-a)*(d-b)-intersection) > threshold:
                break
        else:
            result.append(box)
    return result


def decode(output, size, scaled_size, padding, confidence):
    if output.shape != (1,5,8400) or not np.isfinite(output).all():
        raise ValueError('YOLOv8 단일 클래스 출력 형식/픽셀 오류')
    width,height = size
    sx,sy = scaled_size[0]/width,scaled_size[1]/height
    px,py = padding
    boxes = []
    for cx,cy,w,h,score in output[0].T:
        if not confidence <= score <= 1 or w <= 0 or h <= 0:
            continue
        box = [max(0,math.floor((cx-w/2-px)/sx)),max(0,math.floor((cy-h/2-py)/sy)),
               min(width,math.ceil((cx+w/2-px)/sx)),min(height,math.ceil((cy+h/2-py)/sy))]
        if box[2]-box[0] >= 8 and box[3]-box[1] >= 8:
            boxes.append({'box':box,'confidence':float(score)})
    return suppress(boxes)


def detect(image, path, confidence):
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads=2
    session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    inputs=session.get_inputs()
    if len(inputs)!=1 or inputs[0].shape != [1,3,640,640] or inputs[0].type != 'tensor(float)':
        raise ValueError('검출기 입력은 float32 [1,3,640,640]이어야 합니다.')
    scale=min(640/image.width,640/image.height)
    size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    padding=((640-size[0])//2,(640-size[1])//2)
    letterbox=Image.new('RGB',(640,640),(114,114,114))
    letterbox.paste(image.convert('RGB').resize(size,Image.Resampling.BILINEAR),padding)
    tensor=np.asarray(letterbox,dtype=np.float32).transpose(2,0,1)[None]/255
    outputs=session.run(None,{inputs[0].name:tensor})
    if len(outputs)!=1:
        raise ValueError('단일 출력 검출기만 지원합니다.')
    return decode(outputs[0],image.size,size,padding,confidence)


def refine_regions(image, regions, regenerate, resolution=1024):
    if resolution not in (512,768,1024):
        raise ValueError('지원하지 않는 보정 해상도')
    result=image.convert('RGB').copy()
    union=Image.new('L',image.size)
    records=[]
    for index,region in enumerate(regions):
        x1,y1,x2,y2=region['box']
        margin=max(x2-x1,y2-y1)//3
        cropbox=(max(0,x1-margin),max(0,y1-margin),min(image.width,x2+margin),min(image.height,y2+margin))
        crop=result.crop(cropbox)
        scale=min(resolution/crop.width,resolution/crop.height)
        size=(max(1,round(crop.width*scale)),max(1,round(crop.height*scale)))
        offset=((resolution-size[0])//2,(resolution-size[1])//2)
        canvas=Image.new('RGB',(resolution,resolution));canvas.paste(crop.resize(size,Image.Resampling.LANCZOS),offset)
        refined=regenerate(canvas,index,region)
        if refined.size != (resolution,resolution):
            raise ValueError('디테일러 출력 크기 오류')
        refined=refined.crop((offset[0],offset[1],offset[0]+size[0],offset[1]+size[1])).resize(crop.size,Image.Resampling.LANCZOS)
        mask=Image.new('L',(x2-x1,y2-y1))
        inset=max(1,min(mask.size)//12)
        ImageDraw.Draw(mask).rounded_rectangle((inset,inset,mask.width-inset-1,mask.height-inset-1),radius=min(mask.size)//4,fill=255)
        mask=mask.filter(ImageFilter.GaussianBlur(max(1,inset/2)))
        patch=refined.crop((x1-cropbox[0],y1-cropbox[1],x2-cropbox[0],y2-cropbox[1]))
        result.paste(patch,(x1,y1),mask)
        # A binary union records every pixel that may have changed, including feathered edges.
        union.paste(Image.new('L',mask.size,255),(x1,y1),mask.point(lambda value:255 if value else 0))
        records.append(dict(region,crop=list(cropbox)))
    return result,union,records
