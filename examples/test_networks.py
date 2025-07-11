import numpy as np
# from auv_track_launcher.policys.drqv2 import RandomShiftsAug
from auv_track_launcher.networks.rgb_net import Encoder, RandomShiftsAug
#
# random_image = np.random.uniform(-3, 3, size=(3, 224, 224)).astype(np.float32)
#
# x = RandomShiftsAug(random_image)

import torch
import torch.nn as nn
import torchvision.models as models
model = models.resnet18('IMAGENET1K_V1').to('cuda')
model_1 = models.resnet50('IMAGENET1K_V1').to('cuda')
feature_extractor = nn.Sequential(*list(model.children())[:-1])
print('end')