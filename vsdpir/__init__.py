import numpy as np
import os
import torch
import vapoursynth as vs
from .network_unet import UNetRes
from .utils_model import test_mode

vs_api_below4 = vs.__api_version__.api_major < 4


def DPIR(clip: vs.VideoNode, strength: float=None, task: str='denoise', device_type: str='cuda', device_index: int=0, fp16: bool=False) -> vs.VideoNode:
    '''
    DPIR: Deep Plug-and-Play Image Restoration

    Parameters:
        clip: Clip to process. Only planar format with float sample type of 32 bit depth is supported.

        strength: Strength for deblocking or denoising. Must be greater than 0. Defaults to 50.0 for 'deblock' task, 5.0 for 'denoise' task.

        task: Task to perform. Must be 'deblock' or 'denoise'.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('DPIR: This is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('DPIR: Only RGBS format is supported')

    if strength is not None and strength <= 0:
        raise vs.Error('DPIR: strength must be greater than 0')

    task = task.lower()
    device_type = device_type.lower()

    if task not in ['deblock', 'denoise']:
        raise vs.Error("DPIR: task must be 'deblock' or 'denoise'")

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("DPIR: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('DPIR: CUDA is not available')

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if task == 'deblock':
        if strength is None:
            strength = 50.0
        strength /= 100
        model_name = 'drunet_deblocking_color.pth'
    else:
        if strength is None:
            strength = 5.0
        strength /= 255
        model_name = 'drunet_color.pth'

    model_path = os.path.join(os.path.dirname(__file__), model_name)

    model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model.to(device)
    if fp16:
        model.half()

    noise_level_map = torch.FloatTensor([strength]).repeat(1, 1, clip.height, clip.width)

    def dpir(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img_L = frame_to_tensor(f)
        img_L = torch.cat((img_L, noise_level_map), dim=1)
        img_L = img_L.to(device)
        if fp16:
            img_L = img_L.half()

        with torch.no_grad():
            if img_L.size(2) % 8 == 0 and img_L.size(3) % 8 == 0:
                img_E = model(img_L)
            else:
                img_E = test_mode(model, img_L, refield=64, mode=5)

        return tensor_to_frame(img_E, f)

    return clip.std.ModifyFrame(clips=clip, selector=dpir)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f.get_read_array(plane) if vs_api_below4 else f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    fout = f.copy()
    for plane in range(fout.format.num_planes):
        np.copyto(np.asarray(fout.get_write_array(plane) if vs_api_below4 else fout[plane]), arr[plane, :, :])
    return fout
