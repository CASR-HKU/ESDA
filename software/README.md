# Software model training

## Dataset 

The dataset is available at [eye-tracking-challenge](https://www.kaggle.com/competitions/event-based-eye-tracking-ais2024/data). 
Please download the dataset and put it in the `event_data` folder.

The dataset is organized as follows:
```
ESDA
--software
----event_data
------train
--------1-2
--------1-3
...
------test
--------1-1
--------2-2

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
python main.py --config_file=<path-to-config-file> --mlflow_path <path-to-float32-result-folder>
python main.py --config_file=<path-to-config-file> --mlflow_path <path-to-int8-result-folder> --shift_bit <shift-bit>  --bias_bit <bias-bit> --load <path-to-float32-model>                  
```

For example, if you want to retrain MobileNetV2, you can run the following command:
```bash
cd software
python main.py --config_file=configs/float32/MobileNetV2.json --mlflow_path ../eventNet/checkpoint/exp_float32/MobileNetV2
python main.py --config_file=configs/int8/MobileNetV2.json --mlflow_path ../eventNet/checkpoint/exp_int8/MobileNetV2 --shift_bit 16 --bias_bit 16 --load [path to float32 model]                  
```
, where the `path to float32 model` is the float32 model is saved in `../eventNet/checkpoint/exp_float32/MobileNetV2`. 
The final int8 model will be saved in `../eventNet/checkpoint/exp_int8/MobileNetV2` folder.

## Integer Inference

After obtaining int8 model, before hardware synthesis, you need to generate the integer model. Assuming you are currently in the `software` folder, you can run the following command:

```bash
python int_inference.py --config_file=configs/int8/MobileNetV2.json --checkpoint=[path to int8 model] --shift_bit 16 --bias_bit 16 --int_folder ../eventNet/model/MobileNetV2
```
The **path to int8 model** is the quantized model trained in the previous step. The generated int8 model will be saved in `int/MobileNetV2` folder.


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
