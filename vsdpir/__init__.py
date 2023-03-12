from __future__ import annotations

import math
import os
from dataclasses import dataclass
from threading import Lock

import numpy as np
import tensorrt
import torch
import torch.nn.functional as F
import vapoursynth as vs
from functorch.compile import memory_efficient_fusion
from torch_tensorrt.fx import LowerSetting
from torch_tensorrt.fx.lower import Lowerer
from torch_tensorrt.fx.utils import LowerPrecision
from vsutil import fallback

from .network_unet import UNetRes

__version__ = "3.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

package_dir = os.path.dirname(os.path.realpath(__file__))


class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class CUDAGraphs:
        graph: list[torch.cuda.CUDAGraph]
        static_input: list[torch.Tensor]
        static_output: list[torch.Tensor]

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


@torch.inference_mode()
def dpir(
    clip: vs.VideoNode,
    device_index: int | None = None,
    num_streams: int = 1,
    nvfuser: bool = False,
    cuda_graphs: bool = False,
    trt: bool = False,
    trt_max_workspace_size: int = 1 << 30,
    trt_cache_path: str = package_dir,
    task: str = "deblock",
    strength: float | vs.VideoNode | None = None,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 8,
) -> vs.VideoNode:
    """Deep Plug-and-Play Image Restoration

    :param clip:                    Clip to process. Only RGBH/RGBS/GRAYH/GRAYS formats are supported. RGBH/GRAYH
                                    performs inference in FP16 mode while RGBS/GRAYS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param nvfuser:                 Enable fusion through nvFuser. Not allowed in TensorRT. (experimental)
    :param cuda_graphs:             Use CUDA Graphs to remove CPU overhead associated with launching CUDA kernels
                                    sequentially. Not allowed in TensorRT.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_max_workspace_size:  Maximum workspace size for TensorRT engine.
    :param trt_cache_path:          Path for TensorRT engine file. Engine will be cached when it's built for the first
                                    time. Note each engine is created for specific settings such as model path/name,
                                    precision, workspace etc, and specific GPUs and it's not portable.
    :param task:                    Task to perform. Must be 'deblock' or 'denoise'.
    :param strength:                Strength for deblocking/denoising.
                                    Defaults to 50.0 for 'deblock', 5.0 for 'denoise'.
                                    Also accepts a GRAY8/GRAYH/GRAYS clip for varying strength.
    :param tile_w:                  Tile width. As too large images result in the out of GPU memory issue, so this tile
                                    option will first crop input images into tiles, and then process each of them.
                                    Finally, they will be merged into one image. 0 denotes for do not use tile.
    :param tile_h:                  Tile height.
    :param tile_pad:                Pad size for each tile, to remove border artifacts.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("dpir: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS, vs.GRAYH, vs.GRAYS]:
        raise vs.Error("dpir: only RGBH/RGBS/GRAYH/GRAYS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("dpir: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("dpir: num_streams must be at least 1")

    if num_streams > vs.core.num_threads:
        raise vs.Error("dpir: setting num_streams greater than `core.num_threads` is useless")

    if trt:
        if nvfuser:
            raise vs.Error("dpir: nvfuser and trt are mutually exclusive")

        if cuda_graphs:
            raise vs.Error("dpir: cuda_graphs and trt are mutually exclusive")

    task = task.lower()

    if task not in ["deblock", "denoise"]:
        raise vs.Error("dpir: task must be 'deblock' or 'denoise'")

    if isinstance(strength, vs.VideoNode):
        if strength.format.id not in [vs.GRAY8, vs.GRAYH, vs.GRAYS]:
            raise vs.Error("dpir: strength must be of GRAY8/GRAYH/GRAYS format")

        if strength.width != clip.width or strength.height != clip.height or strength.num_frames != clip.num_frames:
            raise vs.Error("dpir: strength must have the same dimensions and number of frames as main clip")

    if os.path.getsize(os.path.join(package_dir, "drunet_color.pth")) == 0:
        raise vs.Error("dpir: model files have not been downloaded. run 'python -m vsdpir' first")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    if fp16:
        torch.set_default_tensor_type(torch.HalfTensor)

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    color_or_gray = "color" if clip.format.color_family == vs.RGB else "gray"
    noise_format = vs.GRAYH if fp16 else vs.GRAYS

    if task == "deblock":
        model_name = f"drunet_deblocking_{color_or_gray}.pth"

        if isinstance(strength, vs.VideoNode):
            noise = strength.std.Expr("x 100 /", format=noise_format)
        else:
            noise = clip.std.BlankClip(format=noise_format, color=fallback(strength, 50.0) / 100, keep=True)
    else:
        model_name = f"drunet_{color_or_gray}.pth"

        if isinstance(strength, vs.VideoNode):
            noise = strength.std.Expr("x 255 /", format=noise_format)
        else:
            noise = clip.std.BlankClip(format=noise_format, color=fallback(strength, 5.0) / 255, keep=True)

    model_path = os.path.join(package_dir, model_name)

    module = UNetRes(in_nc=clip.format.num_planes + 1, out_nc=clip.format.num_planes)
    module.load_state_dict(torch.load(model_path, map_location="cpu"))
    module.eval().to(device, memory_format=torch.channels_last)

    if tile_w > 0 and tile_h > 0:
        pad_w = math.ceil(min(tile_w + 2 * tile_pad, clip.width) / 8) * 8
        pad_h = math.ceil(min(tile_h + 2 * tile_pad, clip.height) / 8) * 8
    else:
        pad_w = math.ceil(clip.width / 8) * 8
        pad_h = math.ceil(clip.height / 8) * 8

    if nvfuser:
        module = memory_efficient_fusion(module)

    if cuda_graphs:
        graph: list[torch.cuda.CUDAGraph] = []
        static_input: list[torch.Tensor] = []
        static_output: list[torch.Tensor] = []

        for i in range(num_streams):
            static_input.append(
                torch.zeros((1, clip.format.num_planes + 1, pad_h, pad_w), device=device).to(
                    memory_format=torch.channels_last
                )
            )

            torch.cuda.synchronize(device=device)
            stream[i].wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(stream[i]):
                module(static_input[i])
            torch.cuda.current_stream(device=device).wait_stream(stream[i])
            torch.cuda.synchronize(device=device)

            graph.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graph[i], stream=stream[i]):
                static_output.append(module(static_input[i]))

        backend = Backend.CUDAGraphs(graph, static_input, static_output)
    elif trt:
        device_name = torch.cuda.get_device_name(device)
        trt_version = tensorrt.__version__
        dimensions = f"{pad_w}x{pad_h}"
        precision = "fp16" if fp16 else "fp32"
        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_path),
            (
                f"{model_name}"
                + f"_{device_name}"
                + f"_trt-{trt_version}"
                + f"_{dimensions}"
                + f"_{precision}"
                + f"_workspace-{trt_max_workspace_size}"
                + ".pt"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            lower_setting = LowerSetting(
                lower_precision=LowerPrecision.FP16 if fp16 else LowerPrecision.FP32,
                min_acc_module_size=1,
                max_workspace_size=trt_max_workspace_size,
                dynamic_batch=False,
                tactic_sources=1 << int(tensorrt.TacticSource.EDGE_MASK_CONVOLUTIONS)
                | 1 << int(tensorrt.TacticSource.JIT_CONVOLUTIONS),
            )
            lowerer = Lowerer.create(lower_setting=lower_setting)
            module = lowerer(
                module,
                [
                    torch.zeros((1, clip.format.num_planes + 1, pad_h, pad_w), device=device).to(
                        memory_format=torch.channels_last
                    )
                ],
            )
            torch.save(module, trt_engine_path)

        del module
        torch.cuda.empty_cache()
        module = [torch.load(trt_engine_path) for _ in range(num_streams)]
        backend = Backend.TensorRT(module)
    else:
        backend = Backend.Eager(module)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img = frame_to_tensor(f[0], device).clamp(0.0, 1.0)
            noise_level_map = frame_to_tensor(f[1], device)
            img = torch.cat((img, noise_level_map), dim=1)

            if tile_w > 0 and tile_h > 0:
                output = tile_process(img, tile_w, tile_h, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "reflect")

                if cuda_graphs:
                    static_input[local_index].copy_(img)
                    graph[local_index].replay()
                    output = static_output[local_index]
                elif trt:
                    output = module[local_index](img)
                else:
                    output = module(img)

                output = output[:, :, :h, :w]

            return tensor_to_frame(output, f[0].copy())

    return clip.std.FrameEval(lambda n: clip.std.ModifyFrame([clip, noise], inference), clip_src=[clip, noise])


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def tile_process(
    img: torch.Tensor,
    tile_w: int,
    tile_h: int,
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Eager | Backend.CUDAGraphs | Backend.TensorRT,
    index: int,
) -> torch.Tensor:
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

            h, w = input_tile.shape[2:]
            mode = "reflect" if pad_w - w < w and pad_h - h < h else "replicate"
            input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), mode)

            # process tile
            if isinstance(backend, Backend.CUDAGraphs):
                backend.static_input[index].copy_(input_tile)
                backend.graph[index].replay()
                output_tile = backend.static_output[index]
            elif isinstance(backend, Backend.TensorRT):
                output_tile = backend.module[index](input_tile)
            else:
                output_tile = backend.module(input_tile)

            output_tile = output_tile[:, :, :h, :w]

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
