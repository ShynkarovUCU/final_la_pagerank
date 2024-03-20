---
license: apache-2.0
---

MobileNet V2 model from Torchvision fine-tuned for FOOD101 dataset. Checkpoint trained for 30 epoches using https://github.com/AlexKoff88/mobilenetv2_food101.

Top-1 accuracy is 76.3% but one can do better.

The main intend is to use it in samples and demos for model optimization. Here is the advantages:
- FOOD101 can automatically downloaded without registration and SMS.
- It is quite representative to reflect the real world scenarios.
- MobileNet v2 is easy to train and lightweight model which is also representative and used in many public benchmarks.

Here is the code to load the checkpoint in PyTorch:

```python
import sys
import os

import torch
import torch.nn as nn
import torchvision.models as models

FOOD101_CLASSES = 101

def fix_names(state_dict):
    state_dict = {key.replace('module.', ''): value for (key, value) in state_dict.items()}
    return state_dict

model = models.mobilenet_v2(num_classes=FOOD101_CLASSES)  

if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        

        checkpoint = torch.load(checkpoint_path)
        weights = fix_names(checkpoint['state_dict'])
        model.load_state_dict(weights)

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_path, checkpoint['epoch']))
```