

This is the code for the paper "xxx"

The whole pipeline contains three parts
1. Software model training and evaluation
2. Hardware configuration optimization
3. Hardware generation and evaluation


## Installation



### SCIPOPT


### Python

Assuming you have download the anaconda. You can create a new environment by:

```bash
conda create -n esda python
conda activate esda
```

Then install the required packages by:

**Important:** Make sure `$SCIPOPTDIR` is set up correctly.

```bash
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102
pip3 install -r requirements.txt
pip3 install gpkit pyscipopt
```
Finally you need to install the Minkowski Engine by:


```bash
```







## Software model training and evaluation

### 1. Dataset preparation

Download the dataset from the following link and put them in the folder `data/` respectively.



### 2. Software model training

(1) Training float32 model using the following commands

```bash
python main.py --settings_file=<path-to-float32-config-file> -s <path-to-result-folder>
```
And the model will be saved in your <path to result folder>


For example, after downloading the dataset, you can conduct training using the following commands

```bash
python main.py --settings_file=config/config_AE/float32/ASL_2929.yaml -s exp_float32/ASL_2929
python main.py --settings_file=config/config_AE/float32/ASL_w0p5.yaml -s exp_float32/ASL_w0p5
python main.py --settings_file=config/config_AE/float32/DVS_1890.yaml -s exp_float32/DVS_1890
python main.py --settings_file=config/config_AE/float32/DVS_w0p5.yaml -s exp_float32/DVS_w0p5
python main.py --settings_file=config/config_AE/float32/NMNIST.yaml -s exp_float32/NMNIST
python main.py --settings_file=config/config_AE/float32/Roshambo.yaml -s exp_float32/Roshambo
python main.py --settings_file=config/config_AE/float32/NCal_2751.yaml -s exp_float32/NCal_2751
python main.py --settings_file=config/config_AE/float32/NCal_w0p5.yaml -s exp_float32/NCal_w0p5
```

(2) Training int8 model using the following commands

After training with float32 model, you can train the int8 model using the following commands.
```bash
python main.py --settings_file=<path-to-float32-config-file> -s <path-to-result-folder> --load <path-to-float32-model> --shift_bit <shift-bit> --fixBN_ratio <fixBN-ratio>
```
And the model will be saved in your <path to result folder>.

For example, assuming you have trained the float32 model above, you can train the int8 model using the following commands:

```bash
python main.py --settings_file=config/config_AE/int8/ASL_2929.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/ASL_2929 --load exp_float32/ASL_2929/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/ASL_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/ASL_w0p5 --load exp_float32/ASL_w0p5/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/DVS_1890.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/DVS_1890 --load exp_float32/DVS_1890/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/DVS_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/DVS_w0p5 --load exp_float32/DVS_w0p5/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/NMNIST.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NMNIST --load exp_float32/NMNIST/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/Roshambo.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/Roshambo --load exp_float32/Roshambo/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/NCal_2751.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NCal_2751 --load exp_float32/NCal_2751/ckpt.best.pth.tar --shift_bit 32
python main.py --settings_file=config/config_AE/int8/NCal_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NCal_w0p5 --load exp_float32/NCal_w0p5/ckpt.best.pth.tar --shift_bit 32
```


### 3. Int8 model inference





## Hardware configuration optimization

After obtaining the int8 model, the next step is to optimize the hardware configuration based on your model architecture and the dataset.


```bash
cd optimization

```

Assuming you have generate 
