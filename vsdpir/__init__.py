import math
import numpy as np
import os
import torch
import vapoursynth as vs
from .network_unet import UNetRes
from .utils_model import test_mode

vs_api_below4 = vs.__api_version__.api_major < 4


def DPIR(clip: vs.VideoNode, strength: float=None, task: str='denoise', tile_x: int=0, tile_y: int=0, tile_pad: int=0,
         device_type: str='cuda', device_index: int=0, fp16: bool=False) -> vs.VideoNode:
    '''
    DPIR: Deep Plug-and-Play Image Restoration

    Parameters:
        clip: Clip to process. Only planar RGB and Gray formats with float sample type of 32 bit depth are supported.

        strength: Strength for deblocking or denoising. Must be greater than 0. Defaults to 50.0 for 'deblock' task, 5.0 for 'denoise' task.

        task: Task to perform. Must be 'deblock' or 'denoise'.

        tile_x, tile_y: Tile width and height respectively, 0 for no tiling.
            It's recommended that the input's width and height is divisible by the tile's width and height respectively.
            Set it to the maximum value that your GPU supports to reduce its impact on the output.

        tile_pad: Tile padding.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('DPIR: This is not a clip')

    if clip.format.id not in [vs.RGBS, vs.GRAYS]:
        raise vs.Error('DPIR: Only RGBS and GRAYS formats are supported')

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

    is_rgb = clip.format.color_family == vs.RGB

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if task == 'deblock':
        if strength is None:
            strength = 50.0
        strength /= 100
        model_name = 'drunet_deblocking_color.pth' if is_rgb else 'drunet_deblocking_grayscale.pth'
    else:
        if strength is None:
            strength = 5.0
        strength /= 255
        model_name = 'drunet_color.pth' if is_rgb else 'drunet_gray.pth'

    model_path = os.path.join(os.path.dirname(__file__), model_name)

    model = UNetRes(in_nc=4 if is_rgb else 2, out_nc=3 if is_rgb else 1)
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
            if tile_x > 0 and tile_y > 0:
                img_E = tile_process(img_L, tile_x, tile_y, tile_pad, model)
            elif img_L.size(2) % 8 == 0 and img_L.size(3) % 8 == 0:
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


def tile_process(img: torch.Tensor, tile_x: int, tile_y: int, tile_pad: int, model: UNetRes) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel - 1, height, width)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_x)
    tiles_y = math.ceil(height / tile_y)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_x
            ofs_y = y * tile_y

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_x, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_y, height)

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
                output_tile = test_mode(model, input_tile, refield=64, mode=5)

            # output tile area on total image
            output_start_x = input_start_x
            output_end_x = input_end_x
            output_start_y = input_start_y
            output_end_y = input_end_y

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad)
            output_end_x_tile = output_start_x_tile + input_tile_width
            output_start_y_tile = (input_start_y - input_start_y_pad)
            output_end_y_tile = output_start_y_tile + input_tile_height

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

    return output
