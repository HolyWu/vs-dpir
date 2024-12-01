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

__version__ = "4.2.0"

os.environ["CI_BUILD"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class Backend:
    @dataclass
    class Torch:
        module: torch.nn.Module

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


@torch.inference_mode()
def dpir(
    clip: vs.VideoNode,
    device_index: int = 0,
    num_streams: int = 1,
    num_batches: int = 1,
    task: str = "deblock",
    strength: float | vs.VideoNode | None = None,
    tile: list[int] = [0, 0],
    tile_pad: int = 8,
    trt: bool = False,
    trt_static_shape: bool = True,
    trt_min_shape: list[int] = [128, 128],
    trt_opt_shape: list[int] = [1920, 1080],
    trt_max_shape: list[int] = [1920, 1080],
    trt_debug: bool = False,
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Deep Plug-and-Play Image Restoration

    :param clip:                    Clip to process. Only RGBH/RGBS/GRAYH/GRAYS formats are supported. RGBH/GRAYH
                                    perform inference in FP16 mode while RGBS/GRAYS perform inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param num_batches:             Batch of frames per inference to perform.
    :param task:                    Task to perform. Must be 'deblock' or 'denoise'.
    :param strength:                Strength for deblocking/denoising.
                                    Defaults to 50.0 for 'deblock', 5.0 for 'denoise'.
                                    Also accepts a clip of GRAY format for varying strength.
    :param tile:                    Tile width and height. As too large images result in the out of GPU memory issue, so
                                    this tile option will first crop input images into tiles, and then process each of
                                    them. Finally, they will be merged into one image. 0 denotes for do not use tile.
    :param tile_pad:                Pad size for each tile, to remove border artifacts.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_static_shape:        Build with static or dynamic shapes.
    :param trt_min_shape:           Min size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_opt_shape:           Opt size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_max_shape:           Max size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                    to run kernels in parallel if the network contains ops that can run in parallel,
                                    with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                    (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                    for more optimization options. Valid values include integers from 0 to the maximum
                                    optimization level, which is currently 5. (default is 3)
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

    if num_batches < 1:
        raise vs.Error("dpir: num_batches must be at least 1")

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

    if not trt_static_shape:
        if not isinstance(trt_min_shape, list) or len(trt_min_shape) != 2:
            raise vs.Error("dpir: trt_min_shape must be a list with 2 items")

        if any(trt_min_shape[i] < 1 for i in range(2)):
            raise vs.Error("dpir: trt_min_shape must be at least 1")

        if not isinstance(trt_opt_shape, list) or len(trt_opt_shape) != 2:
            raise vs.Error("dpir: trt_opt_shape must be a list with 2 items")

        if any(trt_opt_shape[i] < 1 for i in range(2)):
            raise vs.Error("dpir: trt_opt_shape must be at least 1")

        if not isinstance(trt_max_shape, list) or len(trt_max_shape) != 2:
            raise vs.Error("dpir: trt_max_shape must be a list with 2 items")

        if any(trt_max_shape[i] < 1 for i in range(2)):
            raise vs.Error("dpir: trt_max_shape must be at least 1")

        if any(trt_min_shape[i] >= trt_max_shape[i] for i in range(2)):
            raise vs.Error("dpir: trt_min_shape must be less than trt_max_shape")

    if os.path.getsize(os.path.join(model_dir, "drunet_color.pth")) == 0:
        raise vs.Error("dpir: model files have not been downloaded. run 'python -m vsdpir' first")

    torch.set_float32_matmul_precision("high")

    color_or_gray = "color" if clip.format.color_family == vs.RGB else "gray"
    in_nc = clip.format.num_planes + 1

    fp16 = clip.format.bits_per_sample == 16
    if fp16:
        dtype = torch.half
        noise_format = vs.GRAYH
    else:
        dtype = torch.float
        noise_format = vs.GRAYS

    device = torch.device("cuda", device_index)

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

    if all(t > 0 for t in tile):
        pad_w = math.ceil(min(tile[0] + 2 * tile_pad, clip.width) / 8) * 8
        pad_h = math.ceil(min(tile[1] + 2 * tile_pad, clip.height) / 8) * 8
    else:
        pad_w = math.ceil(clip.width / 8) * 8
        pad_h = math.ceil(clip.height / 8) * 8

    if trt:
        import tensorrt
        import torch_tensorrt

        if trt_static_shape:
            dimensions = f"{pad_w}x{pad_h}"
        else:
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
                + f"_batch-{num_batches}"
                + f"_{dimensions}"
                + f"_{'fp16' if fp16 else 'fp32'}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                + ".ts"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            module = init_module(model_name, in_nc, device, dtype)

            example_inputs = (torch.zeros((num_batches, in_nc, pad_h, pad_w), dtype=dtype, device=device),)

            if trt_static_shape:
                dynamic_shapes = None

                inputs = example_inputs
            else:
                trt_min_shape.reverse()
                trt_opt_shape.reverse()
                trt_max_shape.reverse()

                _height = torch.export.Dim("height", min=trt_min_shape[0] // 8, max=trt_max_shape[0] // 8)
                _width = torch.export.Dim("width", min=trt_min_shape[1] // 8, max=trt_max_shape[1] // 8)
                dim_height = _height * 8
                dim_width = _width * 8
                dynamic_shapes = {"x0": {2: dim_height, 3: dim_width}}

                inputs = [
                    torch_tensorrt.Input(
                        min_shape=[num_batches, in_nc] + trt_min_shape,
                        opt_shape=[num_batches, in_nc] + trt_opt_shape,
                        max_shape=[num_batches, in_nc] + trt_max_shape,
                        dtype=dtype,
                    )
                ]

            exported_program = torch.export.export(module, example_inputs, dynamic_shapes=dynamic_shapes)

            module = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs,
                device=device,
                enabled_precisions={dtype},
                debug=trt_debug,
                num_avg_timing_iters=4,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
            )

            torch_tensorrt.save(module, trt_engine_path, output_format="torchscript", inputs=example_inputs)

        module = [torch.jit.load(trt_engine_path).eval() for _ in range(num_streams)]
        backend = Backend.TensorRT(module)
    else:
        module = init_module(model_name, in_nc, device, dtype)
        backend = Backend.Torch(module)

    index = -1
    index_lock = Lock()

    inf_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    f2t_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    t2f_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]

    inf_stream_locks = [Lock() for _ in range(num_streams)]
    f2t_stream_locks = [Lock() for _ in range(num_streams)]
    t2f_stream_locks = [Lock() for _ in range(num_streams)]

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with f2t_stream_locks[local_index], torch.cuda.stream(f2t_streams[local_index]):
            img = torch.stack([frame_to_tensor(f[i], device) for i in range(num_batches)]).clamp(0.0, 1.0)
            noise_level_map = torch.stack([frame_to_tensor(f[i + num_batches], device) for i in range(num_batches)])
            img = torch.cat([img, noise_level_map], dim=1)

            f2t_streams[local_index].synchronize()

        with inf_stream_locks[local_index], torch.cuda.stream(inf_streams[local_index]):
            if all(t > 0 for t in tile):
                output = tile_process(img, tile, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                if need_pad := pad_w - w > 0 or pad_h - h > 0:
                    img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "replicate")

                if trt:
                    output = module[local_index](img)
                else:
                    output = module(img)

                if need_pad:
                    output = output[:, :, :h, :w]

            inf_streams[local_index].synchronize()

        with t2f_stream_locks[local_index], torch.cuda.stream(t2f_streams[local_index]):
            frame = tensor_to_frame(output[0], f[0].copy(), t2f_streams[local_index])
            for i in range(1, num_batches):
                frame.props[f"vsdpir_batch_frame{i}"] = tensor_to_frame(
                    output[i], f[0].copy(), t2f_streams[local_index]
                )
            return frame

    if (pad := (num_batches - clip.num_frames % num_batches) % num_batches) > 0:
        clip = clip.std.DuplicateFrames([clip.num_frames - 1] * pad)
        noise = noise.std.DuplicateFrames([noise.num_frames - 1] * pad)

    clips = [clip[i::num_batches] for i in range(num_batches)] + [noise[i::num_batches] for i in range(num_batches)]

    outputs = [clips[0].std.FrameEval(lambda n: clips[0].std.ModifyFrame(clips, inference), clip_src=clips)]
    for i in range(1, num_batches):
        outputs.append(outputs[0].std.PropToClip(f"vsdpir_batch_frame{i}"))

    output = vs.core.std.Interleave(outputs)
    if pad > 0:
        output = output[:-pad]
    return output


def init_module(model_name: str, in_nc: int, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    state_dict = torch.load(os.path.join(model_dir, model_name), map_location="cpu", weights_only=True)

    with torch.device("meta"):
        module = UNetRes(in_nc=in_nc, out_nc=in_nc - 1)
    module.load_state_dict(state_dict, assign=True)
    return module.eval().to(device, dtype)


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    )


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame, stream: torch.cuda.Stream) -> vs.VideoFrame:
    tensor = tensor.detach()
    tensors = [tensor[plane].to("cpu", non_blocking=True) for plane in range(frame.format.num_planes)]

    stream.synchronize()

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), tensors[plane].numpy())
    return frame


def tile_process(
    img: torch.Tensor,
    tile: list[int],
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Torch | Backend.TensorRT,
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
            if need_pad := pad_w - w > 0 or pad_h - h > 0:
                input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), "replicate")

            # process tile
            if isinstance(backend, Backend.TensorRT):
                output_tile = backend.module[index](input_tile)
            else:
                output_tile = backend.module(input_tile)

            if need_pad:
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
