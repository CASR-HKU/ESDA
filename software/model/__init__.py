from .BaselineEyeTrackingModel import CNN_RNN
from .mobilenet_standard import MobileNetStandard
from .resnet_standard import resnet18_standard, resnet34_standard, resnet50_standard
from .UNetStandard import UNetStandard
try:
    from .mobilenet_submanifold import MobileNetSubmanifold
    from .submanifoldModel import CNN_RNN_Submanifold
    from .resnet_submanifold import resnet18_submanifold, resnet34_submanifold, resnet50_submanifold
    from .UNetSubmanifold import UNetSubmanifold
    from .HAWQ_mobilenetv2 import MobileNetSubmanifoldQuant
except:
    pass
