# DPIR
DPIR function for VapourSynth, based on https://github.com/cszn/DPIR.


## Dependencies
- [NumPy](https://numpy.org/install)
- [ONNX Runtime](https://onnxruntime.ai/). Note that only one of `onnxruntime`, `onnxruntime-gpu` and `onnxruntime-directml` should be installed at a time in any one environment.
- [VapourSynth](http://www.vapoursynth.com/)
- (Optional) [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- (Optional) [cuDNN](https://developer.nvidia.com/cudnn)
- (Optional) [TensorRT](https://developer.nvidia.com/tensorrt)


## Installation
```
pip install --upgrade vsdpir
python -m vsdpir
```


## Usage
```python
from vsdpir import DPIR

ret = DPIR(clip)
```

See `__init__.py` for the description of the parameters.
