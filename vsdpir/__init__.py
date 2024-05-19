from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .network_unet import UNetRes

__version__ = "4.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


@torch.inference_mode()
def dpir(
    clip: vs.VideoNode,
    device_index: int = 0,
    num_streams: int = 1,
    task: str = "deblock",
    strength: float | vs.VideoNode | None = None,
    tile: list[int] = [0, 0],
    tile_pad: int = 8,
    trt: bool = False,
    trt_debug: bool = False,
    trt_min_shape: list[int] = [128, 128],
    trt_opt_shape: list[int] = [1920, 1080],
    trt_max_shape: list[int] = [1920, 1080],
    trt_workspace_size: int = 0,
    trt_int8: bool = False,
    trt_int8_sample_step: int = 120,
    trt_int8_batch_size: int = 1,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Deep Plug-and-Play Image Restoration

    :param clip:                    Clip to process. Only RGBH/RGBS/GRAYH/GRAYS formats are supported. RGBH/GRAYH
                                    perform inference in FP16 mode while RGBS/GRAYS perform inference in FP32 mode,
                                    except `trt_int8=True`.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param task:                    Task to perform. Must be 'deblock' or 'denoise'.
    :param strength:                Strength for deblocking/denoising.
                                    Defaults to 50.0 for 'deblock', 5.0 for 'denoise'.
                                    Also accepts a clip of GRAY format for varying strength.
    :param tile:                    Tile width and height. As too large images result in the out of GPU memory issue, so
                                    this tile option will first crop input images into tiles, and then process each of
                                    them. Finally, they will be merged into one image. 0 denotes for do not use tile.
    :param tile_pad:                Pad size for each tile, to remove border artifacts.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_min_shape:           Min size of dynamic shapes.
    :param trt_opt_shape:           Opt size of dynamic shapes.
    :param trt_max_shape:           Max size of dynamic shapes.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_int8:                Perform inference in INT8 mode using Post Training Quantization (PTQ). Calibration
                                    datasets are sampled from input clip while building the engine.
    :param trt_int8_sample_step:    Interval between sampled frames.
    :param trt_int8_batch_size:     How many samples per batch to load. Calibrate with as large a single batch as
                                    possible. Batch size can affect truncation error and may impact the final result.
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("dpir: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS, vs.GRAYH, vs.GRAYS]:
        raise vs.Error("dpir: only RGBH/RGBS/GRAYH/GRAYS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("dpir: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("dpir: num_streams must be at least 1")

    task = task.lower()

    if task not in ["deblock", "denoise"]:
        raise vs.Error("dpir: task must be 'deblock' or 'denoise'")

    if isinstance(strength, vs.VideoNode):
        if strength.format.color_family != vs.GRAY:
            raise vs.Error("dpir: strength must be of GRAY format")

        if strength.width != clip.width or strength.height != clip.height or strength.num_frames != clip.num_frames:
            raise vs.Error("dpir: strength must have the same dimensions and number of frames as main clip")

    if not isinstance(tile, list) or len(tile) != 2:
        raise vs.Error("dpir: tile must be a list with 2 items")

    if not isinstance(trt_min_shape, list) or len(trt_min_shape) != 2:
        raise vs.Error("dpir: trt_min_shape must be a list with 2 items")

    if not isinstance(trt_opt_shape, list) or len(trt_opt_shape) != 2:
        raise vs.Error("dpir: trt_opt_shape must be a list with 2 items")

    if not isinstance(trt_max_shape, list) or len(trt_max_shape) != 2:
        raise vs.Error("dpir: trt_max_shape must be a list with 2 items")

    if trt_int8 and clip.format.bits_per_sample != 32:
        raise vs.Error("dpir: INT8 mode only supports RGBS/GRAYS formats")

    if trt_int8_sample_step < 1:
        raise vs.Error("dpir: trt_int8_sample_step must be at least 1")

    if os.path.getsize(os.path.join(model_dir, "drunet_color.pth")) == 0:
        raise vs.Error("dpir: model files have not been downloaded. run 'python -m vsdpir' first")

    torch.set_float32_matmul_precision("high")

    color_or_gray = "color" if clip.format.color_family == vs.RGB else "gray"

    fp16 = clip.format.bits_per_sample == 16
    if fp16:
        dtype = torch.half
        noise_format = vs.GRAYH
    else:
        dtype = torch.float
        noise_format = vs.GRAYS

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    if task == "deblock":
        model_name = f"drunet_deblocking_{color_or_gray}.pth"

        if isinstance(strength, vs.VideoNode):
            noise = strength.std.Expr("x 100 /", format=noise_format)
        else:
            noise = clip.std.BlankClip(
                format=noise_format, color=(50.0 if strength is None else strength) / 100, keep=True
            )
    else:
        model_name = f"drunet_{color_or_gray}.pth"

        if isinstance(strength, vs.VideoNode):
            noise = strength.std.Expr("x 255 /", format=noise_format)
        else:
            noise = clip.std.BlankClip(
                format=noise_format, color=(5.0 if strength is None else strength) / 255, keep=True
            )

    module = UNetRes(in_nc=clip.format.num_planes + 1, out_nc=clip.format.num_planes)
    module.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location=device))
    module.eval().to(device)
    if fp16:
        module.half()

    if all([t > 0 for t in tile]):
        pad_w = math.ceil(min(tile[0] + 2 * tile_pad, clip.width) / 8) * 8
        pad_h = math.ceil(min(tile[1] + 2 * tile_pad, clip.height) / 8) * 8
    else:
        pad_w = math.ceil(clip.width / 8) * 8
        pad_h = math.ceil(clip.height / 8) * 8

    if trt:
        import tensorrt
        import torch_tensorrt
        import torch_tensorrt.ts.logging as logging
        from torch.utils.data import DataLoader, Dataset
        from torch_tensorrt.ts.ptq import DataLoaderCalibrator

        class MyDataset(Dataset):
            def __init__(self, clip: vs.VideoNode, device: torch.device) -> None:
                super().__init__()
                self.clip = clip
                self.device = device

            def __getitem__(self, index: int) -> torch.Tensor:
                with self.clip.get_frame(index * trt_int8_sample_step) as f:
                    return frame_to_tensor(f, self.device).clamp(0.0, 1.0)

            def __len__(self) -> int:
                return math.ceil(self.clip.num_frames / trt_int8_sample_step)

        logging.set_reportable_log_level(logging.Level.Debug if trt_debug else logging.Level.Info)
        logging.set_is_colored_output_on(True)

        for i in range(2):
            trt_min_shape[i] = math.ceil(trt_min_shape[i] / 8) * 8
            trt_opt_shape[i] = math.ceil(trt_opt_shape[i] / 8) * 8
            trt_max_shape[i] = math.ceil(trt_max_shape[i] / 8) * 8

        dimensions = (
            f"min-{trt_min_shape[0]}x{trt_min_shape[1]}"
            f"_opt-{trt_opt_shape[0]}x{trt_opt_shape[1]}"
            f"_max-{trt_max_shape[0]}x{trt_max_shape[1]}"
        )

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_{dimensions}"
                + f"_{'int8' if trt_int8 else 'fp16' if fp16 else 'fp32'}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + ".ts"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            inputs = [torch.zeros((1, clip.format.num_planes + 1, 128, 128), dtype=dtype, device=device)]
            module = torch.jit.trace(module, inputs)

            batch_channel = [1, clip.format.num_planes + 1]

            trt_min_shape.reverse()
            trt_opt_shape.reverse()
            trt_max_shape.reverse()

            inputs = [
                torch_tensorrt.Input(
                    min_shape=batch_channel + trt_min_shape,
                    opt_shape=batch_channel + trt_opt_shape,
                    max_shape=batch_channel + trt_max_shape,
                    dtype=dtype,
                )
            ]

            if trt_int8:
                dataset = MyDataset(clip, device)
                dataloader = DataLoader(dataset, batch_size=trt_int8_batch_size)
                calibrator = DataLoaderCalibrator(dataloader, device=device)

            module = torch_tensorrt.compile(
                module,
                ir="ts",
                inputs=inputs,
                enabled_precisions={torch.half, torch.int8} if trt_int8 else {dtype},
                device=torch_tensorrt.Device(gpu_id=device_index),
                workspace_size=trt_workspace_size,
                calibrator=calibrator if trt_int8 else None,
                truncate_long_and_double=True,
                min_block_size=1,
                allow_shape_tensors=True,
            )

            torch.jit.save(module, trt_engine_path)

        module = [torch.jit.load(trt_engine_path) for _ in range(num_streams)]
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
            img = torch.cat((img, noise_level_map)).unsqueeze(0)

            if all([t > 0 for t in tile]):
                output = tile_process(img, tile, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "reflect")

                if trt:
                    output = module[local_index](img)
                else:
                    output = module(img)

                output = output[:, :, :h, :w]

            return tensor_to_frame(output, f[0].copy())

    return clip.std.FrameEval(lambda n: clip.std.ModifyFrame([clip, noise], inference), clip_src=[clip, noise])


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [torch.from_numpy(np.asarray(frame[plane])).to(device) for plane in range(frame.format.num_planes)]
    )


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane])
    return frame


def tile_process(
    img: torch.Tensor,
    tile: list[int],
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Eager | Backend.TensorRT,
    index: int,
) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel - 1, height, width)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile[0])
    tiles_y = math.ceil(height / tile[1])

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile[0]
            ofs_y = y * tile[1]

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile[0], width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile[1], height)

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
            if isinstance(backend, Backend.TensorRT):
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
