# BRU-chainer
Bionodal Root Unit (BRU) implemented with Chainer-v5.  
See: https://arxiv.org/abs/1804.11237

## Usage
### ERU (exponential root unit)
<img src="https://github.com/takyamamoto/BRU_chainer/blob/master/figs/ERU.png" width=40%>

```
>>> import numpy as np
>>> from eru import eru
>>> x = np.array([[-1, 0], [2, -3]], np.float32)
>>> x
array([[-1.,  0.],
       [ 2., -3.]], dtype=float32)
>>> y = eru(x, r=2)
>>> y.data
array([[-0.36466473,  0.5       ],
       [ 2.5       , -0.49752125]], dtype=float32)
```

### ORU (odd root unit)
<img src="https://github.com/takyamamoto/BRU_chainer/blob/master/figs/ORU.png" width=40%>

```
>>> import numpy as np
>>> from oru import oru
>>> x = np.array([[-1, 0], [2, -3]], np.float32)
>>> x
array([[-1.,  0.],
       [ 2., -3.]], dtype=float32)
>>> y = oru(x, r=2)
>>> y.data
array([[-1.236068 ,  0.       ],
       [ 2.       , -2.6055512]], dtype=float32)
```

### Plot functions
```
python plot_bru_functions.py
```
