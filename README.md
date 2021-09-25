# DPIR
DPIR function for VapourSynth, based on https://github.com/cszn/DPIR.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchvision` and `torchaudio` are not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)


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
