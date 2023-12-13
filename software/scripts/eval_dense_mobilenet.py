import torchvision.models as models
import torch
import time

width = 1.0
model = models.mobilenet_v2(pretrained=True, width_mult=width)
input_size = [180, 240]
sample_size = 5
times = []

for idx in range(sample_size):
    begin_time = time.time()
    sample = torch.randn(1, 3, input_size[0], input_size[1])
    model(sample)
    times.append(time.time() - begin_time)

print("Average time: {}".format(sum(times) / len(times)))
