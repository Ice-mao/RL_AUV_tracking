import numpy as np
# from atrl_launcher.policys.drqv2 import RandomShiftsAug
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

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

input_batch = input_batch.to('cuda')

with torch.no_grad():
    output = feature_extractor(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)