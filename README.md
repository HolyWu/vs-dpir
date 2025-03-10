# DPIR
Plug-and-Play Image Restoration with Deep Denoiser Prior, based on https://github.com/cszn/DPIR.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional packages:
- [TensorRT](https://developer.nvidia.com/tensorrt) 10.7.0.post1 or later
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0 or later

To install the latest stable version of PyTorch and Torch-TensorRT, run:
```
pip install -U packaging setuptools wheel
pip install -U torch torchvision torch_tensorrt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.nvidia.com
```


## Installation
```
pip install -U vsdpir
```

If you want to download all models at once, run `python -m vsdpir`. If you prefer to only download the model you
specified at first run, set `auto_download=True` in `dpir()`.


## Usage
```python
from vsdpir import dpir

ret = dpir(clip)
```

See `__init__.py` for the description of the parameters.
