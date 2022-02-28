import math
import os.path as osp
from typing import Optional

import numpy as np
import torch
import vapoursynth as vs

dir_name = osp.dirname(__file__)


def DPIR(
    clip: vs.VideoNode,
    strength: Optional[float] = None,
    task: str = 'denoise',
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 0,
    device_type: str = 'cuda',
    device_index: Optional[int] = None,
    fp16: bool = False,
    trt: bool = False,
    save_trt_model: bool = False,
) -> vs.VideoNode:
    '''
    DPIR: Deep Plug-and-Play Image Restoration

    Parameters:
        clip: Clip to process. Only RGB and GRAY formats with float sample type of 32 bit depth are supported.

        strength: Strength for deblocking or denoising. Must be greater than 0. Defaults to 50.0 for 'deblock' task, 5.0 for 'denoise' task.

        task: Task to perform. Must be 'deblock' or 'denoise'.

        tile_w, tile_h: Tile width and height, respectively. As too large images result in the out of GPU memory issue,
            so this tile option will first crop input images into tiles, and then process each of them.
            Finally, they will be merged into one image. 0 denotes for do not use tile.

        tile_pad: The pad size for each tile, to remove border artifacts.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: Whether to use FP16 precision during inference.

        trt: Use TensorRT model to accelerate inferencing.

        save_trt_model: Save the converted TensorRT model and does no inference. One-frame evaluation is enough.
            Each model can only work with a specific dimension, hence you must save the model first for dimensions which have not been converted.
            Keep in mind that models are not portable across platforms or TensorRT versions and are specific to the exact GPU model they were built on.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('DPIR: this is not a clip')

    if clip.format.id not in [vs.RGBS, vs.GRAYS]:
        raise vs.Error('DPIR: only RGBS and GRAYS formats are supported')

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

    if trt and save_trt_model:
        raise vs.Error('DPIR: both trt and save_trt_model cannot be True at the same time')

    if (trt or save_trt_model) and device_type == 'cpu':
        raise vs.Error('DPIR: TensorRT is not supported for CPU device')

    if osp.getsize(osp.join(dir_name, 'drunet_color.pth')) == 0:
        raise vs.Error("DPIR: model files have not been downloaded. run 'python -m vsdpir' first")

    is_rgb = clip.format.color_family == vs.RGB
    color_or_gray = 'color' if is_rgb else 'gray'

    if tile_w > 0 and tile_h > 0:
        trt_width = (min(tile_w + tile_pad, clip.width) + 7) & ~7
        trt_height = (min(tile_h + tile_pad, clip.height) + 7) & ~7
    else:
        trt_width = (clip.width + 7) & ~7
        trt_height = (clip.height + 7) & ~7

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if task == 'deblock':
        trt_model_name = f'drunet_deblocking_{color_or_gray}_trt_{trt_width}x{trt_height}{"_fp16" if fp16 else ""}.pth'
        model_name = trt_model_name if trt else f'drunet_deblocking_{color_or_gray}.pth'
        if strength is None:
            strength = 50.0
        strength /= 100
        clip = clip.std.Limiter()
    else:
        trt_model_name = f'drunet_{color_or_gray}_trt_{trt_width}x{trt_height}{"_fp16" if fp16 else ""}.pth'
        model_name = trt_model_name if trt else f'drunet_{color_or_gray}.pth'
        if strength is None:
            strength = 5.0
        strength /= 255

    model_path = osp.join(dir_name, model_name)
    trt_model_path = osp.join(dir_name, trt_model_name)

    if trt:
        from torch2trt import TRTModule

        model = TRTModule()
    else:
        from .network_unet import UNetRes

        model = UNetRes(in_nc=4 if is_rgb else 2, out_nc=3 if is_rgb else 1)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model.to(device)
    if fp16:
        model.half()

    if save_trt_model:
        with torch.inference_mode():
            from torch2trt import torch2trt

            x = torch.ones((1, 4 if is_rgb else 2, trt_height, trt_width), dtype=torch.half if fp16 else torch.float, device=device)
            model_trt = torch2trt(model, [x], fp16_mode=fp16)
            torch.save(model_trt.state_dict(), trt_model_path)
            vs.core.log_message(1, f"'{trt_model_path}' saved successfully")
            return clip

    noise_level_map = torch.FloatTensor([strength]).repeat(1, 1, clip.height, clip.width)

    @torch.inference_mode()
    def dpir(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img = frame_to_tensor(f)
        img = torch.cat((img, noise_level_map), dim=1)
        img = img.to(device)
        if fp16:
            img = img.half()

        if tile_w > 0 and tile_h > 0:
            img = tile_process(img, tile_w, tile_h, tile_pad, model)
        elif img.size(2) % 8 == 0 and img.size(3) % 8 == 0:
            img = model(img)
        else:
            img = mod_pad(img, 8, model)

        return tensor_to_frame(img, f.copy())

    return clip.std.ModifyFrame(clips=clip, selector=dpir)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f[plane]), arr[plane, :, :])
    return f


def tile_process(img: torch.Tensor, tile_w: int, tile_h: int, tile_pad: int, model: torch.nn.Module) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel - 1, height, width)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_w)
    tiles_y = math.ceil(height / tile_h)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_w
            ofs_y = y * tile_h

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_w, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_h, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # process tile
            if input_tile.size(2) % 8 == 0 and input_tile.size(3) % 8 == 0:
                output_tile = model(input_tile)
            else:
                output_tile = mod_pad(input_tile, 8, model)

            # output tile area on total image
            output_start_x = input_start_x
            output_end_x = input_end_x
            output_start_y = input_start_y
            output_end_y = input_end_y

            # output tile area without padding
            output_start_x_tile = input_start_x - input_start_x_pad
            output_end_x_tile = output_start_x_tile + input_tile_width
            output_start_y_tile = input_start_y - input_start_y_pad
            output_end_y_tile = output_start_y_tile + input_tile_height

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output


def mod_pad(img: torch.Tensor, modulo: int, model: torch.nn.Module) -> torch.Tensor:
    import torch.nn.functional as F

    mod_pad_h, mod_pad_w = 0, 0
    h, w = img.shape[2:]

    if h % modulo != 0:
        mod_pad_h = modulo - h % modulo

    if w % modulo != 0:
        mod_pad_w = modulo - w % modulo

    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    output = model(img)
    return output[:, :, :h, :w]
