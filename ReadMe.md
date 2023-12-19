

This is the code for the paper "xxx"

The whole pipeline contains three parts
1. Software model training and evaluation
2. Hardware configuration optimization
3. Hardware generation and evaluation


## Installation



### SCIPOPT


### Python

Assuming you have download the [anaconda](https://www.anaconda.com/download). You can create a new environment by:

```bash
conda create -n esda python
conda activate esda
```

Or you can directly download [python 3.8](https://www.python.org/downloads/release/python-380/)


Then install the required packages by:

**Important:** Make sure `$SCIPOPTDIR` is set up correctly.

```bash
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102
pip3 install -r requirements.txt
pip3 install gpkit pyscipopt
```
Finally you need to install the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine):


```bash
```





## Software model training and evaluation

### 1. Dataset preparation

Please refer to [dataset preparation](doc/dataset.md) for the details.



### 2. Software model training

(1) Training float32 model

```bash
python main.py --settings_file=<path-to-float32-config-file> -s <path-to-result-folder>
```
And the model will be saved in your <path to result folder>


For example, after downloading the dataset, you can conduct training using the following commands

```bash
python main.py --settings_file=config/default/float32/ASL_2929.yaml -s exp_float32/ASL_2929
python main.py --settings_file=config/default/float32/ASL_w0p5.yaml -s exp_float32/ASL_w0p5
python main.py --settings_file=config/default/float32/DVS_1890.yaml -s exp_float32/DVS_1890
python main.py --settings_file=config/default/float32/DVS_w0p5.yaml -s exp_float32/DVS_w0p5
python main.py --settings_file=config/default/float32/NMNIST.yaml -s exp_float32/NMNIST
python main.py --settings_file=config/default/float32/Roshambo.yaml -s exp_float32/Roshambo
python main.py --settings_file=config/default/float32/NCal_2751.yaml -s exp_float32/NCal_2751
python main.py --settings_file=config/default/float32/NCal_w0p5.yaml -s exp_float32/NCal_w0p5
```

(2) Training int8 model

After training with float32 model, you can train the int8 model using the following commands.
```bash
python main.py --settings_file=<path-to-float32-config-file> -s <path-to-result-folder> --load <path-to-float32-model> --shift_bit <shift-bit> --fixBN_ratio <fixBN-ratio>
```
And the model will be saved in your <path to result folder>.

For example, assuming you have trained the float32 model above, you can train the int8 model using the following commands:

```bash
python main.py --settings_file=config/default/int8/ASL_2929.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/ASL_2929 --load exp_float32/ASL_2929/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/default/int8/ASL_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/ASL_w0p5 --load exp_float32/ASL_w0p5/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/default/int8/DVS_1890.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/DVS_1890 --load exp_float32/DVS_1890/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/default/int8/DVS_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/DVS_w0p5 --load exp_float32/DVS_w0p5/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/default/int8/NMNIST.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NMNIST --load exp_float32/NMNIST/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/default/int8/Roshambo.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/Roshambo --load exp_float32/Roshambo/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/default/int8/NCal_2751.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NCal_2751 --load exp_float32/NCal_2751/ckpt.best.pth.tar --shift_bit 32
python main.py --settings_file=config/default/int8/NCal_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NCal_w0p5 --load exp_float32/NCal_w0p5/ckpt.best.pth.tar --shift_bit 32
```


### 3. Int8 model inference






## Hardware configuration optimization

### 1. Hardware searching

After obtaining the int8 model, the next step is to optimize the hardware configuration based on your model architecture and the dataset.


```bash
# Assuming you are in the root directory
cd optimization
python 
```

For example, assuming you have generate the corresponding model structures and input data using the commands above.
You will obtain a folder of model structure and feature files

```
EDSA
├── software
├── hardware
├── optimization
├── hw/model
│   ├── ASL_2929_shift16
│   │   ├── model.json
│   │   ├── input_Features.npy
│   │   ├── input_Coordinates.npy
│   │   ├── output_logit.npy
│   │   ...
│   ├── ASL_w0p5_shift16
│   ├── DVS_1890_shift16
│   ├── DVS_w0p5_shift16
│   ├── NCal_2751_shift32
│   ├── NCal_w0p5_shift32
│   ├── NMNIST_shift16
│   ├── Roshambo_shift16
```

You can conduct hardware configuration optimization by the following commands
```bash
# Make sure you are in the root directory
python optimization/eventnet.py --model_path hw/model --hw_path /vol/datastore/EDSA/eventNetHWConfig --model_name DVS_1890_shift16 --hw_name zcu102_80res --results_path hw/DSE
python optimization/eventnet.py --model_path hw/model --hw_path /vol/datastore/EDSA/eventNetHWConfig --model_name DVS_0p5_shift16 --hw_name zcu102_60res --results_path hw/DSE
python optimization/eventnet.py --model_path hw/model --hw_path /vol/datastore/EDSA/eventNetHWConfig --model_name NMNIST_shift16 --hw_name zcu102_60res --results_path hw/DSE
python optimization/eventnet.py --model_path hw/model --hw_path /vol/datastore/EDSA/eventNetHWConfig --model_name ASL_0p5_shift16 --hw_name zcu102_80res --results_path hw/DSE
python optimization/eventnet.py --model_path hw/model --hw_path /vol/datastore/EDSA/eventNetHWConfig --model_name ASL_2929_shift16 --hw_name zcu102_80res --results_path hw/DSE
python optimization/eventnet.py --model_path hw/model --hw_path /vol/datastore/EDSA/eventNetHWConfig --model_name Roshambo_shift16 --hw_name zcu102_80res --results_path hw/DSE
python optimization/eventnet.py --model_path hw/model --hw_path /vol/datastore/EDSA/eventNetHWConfig --model_name NCal_2751_shift32 --hw_name zcu102_80res --results_path hw/DSE
python optimization/eventnet.py --model_path hw/model --hw_path /vol/datastore/EDSA/eventNetHWConfig --model_name NCal_w0p5_shift32_2 --hw_name zcu102_50res --results_path hw/DSE
```

The result

### 2. Project generation


```bash
# Make sure you are in the root directory
cd hardware
```

For example, assuming you generate the DSE result using the commands above, you will generate the config files in the hw/DSE file

```
EDSA
├── software
├── hardware
├── optimization
├── hw
│   ├── model
│   ├── DSE
│   │   ├── ASL_2929_shift16
│   │   ├── model.json
│   │   ├── input_Features.npy
│   │   ├── input_Coordinates.npy
│   │   ├── output_logit.npy
```

Then you can generate the corresponding hardware project by:






## Hardware generation and evaluation

