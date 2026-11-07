"""Exact INT8 ConvRot restoration using the NVRTC runtime bundled with Torch."""
import ctypes as c
import os
from pathlib import Path
import torch

SOURCE=r'''
extern "C" __global__ void decode(const signed char* input, const float* scale,
    unsigned short* output, long long count, int width, int group) {
    __shared__ int values[256];
    int t=threadIdx.x;
    long long i=(long long)blockIdx.x*256+t;
    int value=i<count ? (int)input[i] : 0;
    for(int stride=1;stride<16;stride*=4) {
        int digit=(t/stride)%4;
        int base=(t%32)-digit*stride;
        int a=__shfl_sync(0xffffffffu,value,base);
        int b=__shfl_sync(0xffffffffu,value,base+stride);
        int c=__shfl_sync(0xffffffffu,value,base+2*stride);
        int d=__shfl_sync(0xffffffffu,value,base+3*stride);
        int opposite=__shfl_sync(0xffffffffu,value,base+(3-digit)*stride);
        value=a+b+c+d-2*opposite;
    }
    values[t]=value;
    __syncthreads();
    for(int stride=16;stride<group;stride*=4) {
        int digit=(t/stride)%4;
        int base=t-digit*stride;
        int total=values[base]+values[base+stride]+values[base+2*stride]+values[base+3*stride];
        int next=total-2*values[base+(3-digit)*stride];
        __syncthreads();
        values[t]=next;
        __syncthreads();
    }
    if(i<count) {
        float norm=group==16 ? 0.25f : group==64 ? 0.125f : 0.0625f;
        float value=((float)values[t]*norm)*scale[(unsigned int)i/(unsigned int)width];
        unsigned int bits=__float_as_uint(value);
        output[i]=(unsigned short)((bits+0x7fffu+((bits>>16)&1u))>>16);
    }
}
'''


class Kernel:
    def __init__(self,source=SOURCE):
        folder=Path(torch.__file__).parent/'lib'
        self.directory=os.add_dll_directory(str(folder))
        self.nvrtc=c.CDLL(str(folder/'nvrtc64_120_0.dll'))
        self.driver=c.WinDLL('nvcuda.dll',winmode=0x00000800)
        signatures={
            'nvrtcCreateProgram':[c.POINTER(c.c_void_p),c.c_char_p,c.c_char_p,c.c_int,c.c_void_p,c.c_void_p],
            'nvrtcCompileProgram':[c.c_void_p,c.c_int,c.POINTER(c.c_char_p)],
            'nvrtcGetProgramLogSize':[c.c_void_p,c.POINTER(c.c_size_t)],
            'nvrtcGetProgramLog':[c.c_void_p,c.c_void_p],
            'nvrtcGetPTXSize':[c.c_void_p,c.POINTER(c.c_size_t)],
            'nvrtcGetPTX':[c.c_void_p,c.c_void_p],
            'nvrtcDestroyProgram':[c.POINTER(c.c_void_p)],
        }
        for name,args in signatures.items():getattr(self.nvrtc,name).argtypes=args
        self.driver.cuModuleLoadData.argtypes=[c.POINTER(c.c_void_p),c.c_void_p]
        self.driver.cuModuleGetFunction.argtypes=[c.POINTER(c.c_void_p),c.c_void_p,c.c_char_p]
        self.driver.cuLaunchKernel.argtypes=[c.c_void_p,*([c.c_uint]*7),c.c_void_p,c.POINTER(c.c_void_p),c.c_void_p]
        program=c.c_void_p()
        self.check(self.nvrtc.nvrtcCreateProgram(c.byref(program),source.encode(),b'decode.cu',0,None,None))
        try:
            major,minor=torch.cuda.get_device_capability()
            options=(c.c_char_p*2)(f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--fmad=false')
            code=self.nvrtc.nvrtcCompileProgram(program,2,options)
            size=c.c_size_t();self.check(self.nvrtc.nvrtcGetProgramLogSize(program,c.byref(size)))
            log=c.create_string_buffer(size.value);self.check(self.nvrtc.nvrtcGetProgramLog(program,log))
            if code:raise RuntimeError(log.value.decode())
            self.check(self.nvrtc.nvrtcGetPTXSize(program,c.byref(size)))
            ptx=c.create_string_buffer(size.value);self.check(self.nvrtc.nvrtcGetPTX(program,ptx))
        finally:self.check(self.nvrtc.nvrtcDestroyProgram(c.byref(program)))
        torch.empty(1,device='cuda')
        self.module=c.c_void_p();self.function=c.c_void_p()
        self.check(self.driver.cuModuleLoadData(c.byref(self.module),ptx))
        self.check(self.driver.cuModuleGetFunction(c.byref(self.function),self.module,b'decode'))

    @staticmethod
    def check(code):
        if code:raise RuntimeError('CUDA/NVRTC status '+str(code))

    def __call__(self,raw,scale,group):
        if not (raw.dtype==torch.int8 and raw.ndim==2 and raw.is_cuda and raw.is_contiguous() and 0<raw.numel()<2**32):raise ValueError('Invalid INT8 input')
        if raw.device.index!=torch.cuda.current_device():raise ValueError('Input must use the current CUDA device')
        if not (scale.dtype==torch.float32 and scale.device==raw.device and scale.shape==(raw.shape[0],1) and scale.is_contiguous()):raise ValueError('Invalid INT8 scale shape/dtype/device')
        if not (type(group) is int and group in (16,64,256) and raw.shape[1]%group==0):raise ValueError('Invalid ConvRot group/shape')
        output=torch.empty_like(raw,dtype=torch.bfloat16)
        args=[c.c_void_p(x.data_ptr()) for x in (raw,scale,output)]+[c.c_longlong(raw.numel()),c.c_int(raw.shape[1]),c.c_int(group)]
        pointers=(c.c_void_p*len(args))(*(c.addressof(x) for x in args))
        self.check(self.driver.cuLaunchKernel(self.function,(raw.numel()+255)//256,1,1,256,1,1,0,c.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None))
        return output


_kernel=None
_unavailable=False


def decode_int8(raw,scale,group):
    global _kernel,_unavailable
    if os.name!='nt' or _unavailable:return None
    if _kernel is None:
        try:_kernel=Kernel()
        except (OSError,RuntimeError) as error:
            _unavailable=True
            import warnings
            warnings.warn('Fused INT8 restore unavailable; using Torch: '+str(error),RuntimeWarning)
            return None
    return _kernel(raw,scale,group)
