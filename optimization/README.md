# Hardware Design Optimization

## Hardware config searching
With an int8 model as well as the model configuration, you can search for hardware configuration based on your model architecture, dataset, FPGA board resource (default ZCU102).


```bash
cd optimization
python eventnet.py --model_path <path-of-models-root-folder> --eventNet_path <path-of-hw-config-folder> --model_name  <name-of-the-target-folder> --eventNet_name <name-of-the-target-hw-config> --results_path <path-of-stored-result>
```

Example command and file structure
```bash
#In the root directory
python eventnet.py --model_path ../eventNet/model --eventNet_path ../hardware/HWConfig --model_name MobileNetV2 --eventNet_name zcu102_60res --results_path ../eventNet/DSE
```

```
EDSA
├── software
├── hardware
├── optimization
├── eventNet/model
│   ├── DVS_1890_shift16
│   │   ├── model.json
│   │   ├── input_Features.npy
│   │   ├── input_Coordinates.npy
│   │   ├── output_logit.npy
```

