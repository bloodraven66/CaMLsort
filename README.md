# tonicBurstingPackage

install with
```
pip install git+https://github.com/bloodraven66/tonicBurstingPackage.git
```

Usage
```
>>> import numpy as np
>>> from TVB.tvb import TVB_handler
>>> tvb_handler = TVB_handler("3DNN")
>>> data = np.random.randn(9,900)
>>> output = tvb_handler.predict(data)
```

Supported models - '1CNN',  '3CNN', '3DNN', 'CNNLSTM',  'Seq_CNNLSTM',  'Seq_CNNLSTM_1sec'

Documentation - 

Features
- Get predictions from different pretrained models
- evaluate dataset with metrics
- Train/ Fine tune models
