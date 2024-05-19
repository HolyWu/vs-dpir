# DPIR
Plug-and-Play Image Restoration with Deep Denoiser Prior, based on https://github.com/cszn/DPIR.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.3 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional Python packages:
- [TensorRT](https://developer.nvidia.com/tensorrt/) 10.0.1
- [Torch-TensorRT](https://pytorch.org/TensorRT/)

To install TensorRT, run `pip install tensorrt==10.0.1 tensorrt-cu12_bindings==10.0.1 tensorrt-cu12_libs==10.0.1 --extra-index-url https://pypi.nvidia.com`

To install Torch-TensorRT, Windows users can pip install the whl file on [Releases](https://github.com/HolyWu/vs-dpir/releases). Linux users can run `pip install --pre torch-tensorrt --index-url https://download.pytorch.org/whl/nightly/cu121` (requires PyTorch nightly build).


## Installation
```
pip install -U vsdpir
python -m vsdpir
```


## Usage
```python
from vsdpir import dpir

ret = dpir(clip)
```

See `__init__.py` for the description of the parameters.
