# Software model training

## Dataset 

The dataset is available at [eye-tracking-challenge](https://www.kaggle.com/competitions/event-based-eye-tracking-ais2024/data). 
Please download the dataset and put it in the `event_data` folder.

The dataset is organized as follows:
```
ESDA
├── software
│   ├── event_data
│   │   ├── train
│   │   │   ├── 1-2
│   │   │   ├── 1-3
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── 1-1
│   │   │   ├── 2-2
```

## Checkpoint evaluation

The checkpoints in the paper are available in the `weights` folder. 
You can evaluate the performance of the model by running the following command:
```bash
cd software
python main.py -e --config_file=weights/Table2/SEE-A/cfg.json --checkpoint=weights/Table2/SEE-A/model_best_p10_acc.pth --shift_bit 16 --bias_bit 16
python main.py -e --config_file=weights/Table2/SEE-B/cfg.json --checkpoint=weights/Table2/SEE-B/model_best_p10_acc.pth --shift_bit 16 --bias_bit 16
python main.py -e --config_file=weights/Table2/SEE-C/cfg.json --checkpoint=weights/Table2/SEE-C/model_best_p10_acc.pth --shift_bit 16 --bias_bit 16
python main.py -e --config_file=weights/Table2/SEE-D/cfg.json --checkpoint=weights/Table2/SEE-D/model_best_p10_acc.pth --shift_bit 16 --bias_bit 16
python main.py -e --config_file=weights/Table2/MobileNetV2/cfg.json --checkpoint=weights/Table2/MobileNetV2/model_best_p10_acc.pth --shift_bit 16 --bias_bit 16
```


## Model training
The training follows a 2-step process. You need to first train a float32 model and then quantize it to int8.
The commands are as follows:
```bash
cd software
python main.py --config_file=<path-to-config-file> --mlflow_path <path-to-float32-result-folder> --num_epochs <number-epochs>
python main.py --config_file=<path-to-config-file> --mlflow_path <path-to-int8-result-folder> --shift_bit <shift-bit>  --bias_bit <bias-bit> --load <path-to-float32-model> --num_epochs <number-epochs>                 
```

For example, if you want to retrain MobileNetV2, you can run the following command:
```bash
# Train float32 model
cd software
python main.py --config_file=configs/float32/MobileNetV2.json --mlflow_path ../eventNet/checkpoint/exp_float32/MobileNetV2 --num_epochs 100
```
The float32 model is saved in `../eventNet/checkpoint/exp_float32/MobileNetV2`.
Unfortunately, the `mlflow` library generated subfolder with random string that you have to assign th model path of the float32 model manually.
For example, in my experiment, the architecture of the saved model is like this:
```
ESDA
├── eventNet
│   ├── checkpoint/exp_float32
│   │   ├── MobileNetV2
│   │   │   ├── .trash
│   │   │   ├── 0
│   │   │   ├── 205614538730105817
│   │   │   │   ├── 0be39c6789004cc58a342f39d6f10897
│   │   │   │   │   ├── artifacts
│   │   │   │   │   │   ├── model_best_p10_acc.pth
│   │   │   │   │   │   ├── model_best_p5_acc.pth
│   │   │   │   │   ├── metrics
│   │   │   │   │   ├── params
│   │   │   │   │   ├── tags
│   │   │   │   │   ├── meta.yaml
```
If choosing the model with best_p10 accuray, the model path of float32 is 
`../eventNet/checkpoint/exp_float32/MobileNetV2/205614538730105817/0be39c6789004cc58a342f39d6f10897/artifacts/model_best_p10_acc.pth`

Then the int8 model is trained loading the float32 model:
```bash
# Train int8 model
python main.py --config_file=configs/int8/MobileNetV2.json --mlflow_path ../eventNet/checkpoint/exp_int8/MobileNetV2 --shift_bit 16 --bias_bit 16  --num_epochs 100 --load ../eventNet/checkpoint/exp_float32/MobileNetV2/205614538730105817/0be39c6789004cc58a342f39d6f10897/artifacts/model_best_p10_acc.pth                 
```
The final int8 model will be saved in `../eventNet/checkpoint/exp_int8/MobileNetV2` folder.


## Integer Inference

After obtaining int8 model, before hardware synthesis, you need to generate the integer model. Assuming you are currently in the `software` folder, you can run the following command:

```bash
python int_inference.py --config_file=configs/int8/MobileNetV2.json --checkpoint=[path to int8 model] --shift_bit 16 --bias_bit 16 --int_folder ../eventNet/model/MobileNetV2
```
The `path to int8 model` is the quantized model trained in the previous step. You can find the model path similar to the float32 path.
The generated int8 model will be saved in `int/MobileNetV2` folder.


```
EDSA
├── software
├── hardware
├── optimization
├── eventNet
│   ├── checkpoint
│   │   ├── exp_float32
│   │   ├── exp_int8
│   ├── model
│   │   ├── MobileNetV2
```

After the steps above, you can refer to [optimization](../optimization/README.md) to conduct the following steps.
