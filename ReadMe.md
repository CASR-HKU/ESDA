# ESDA for Eye Tracking Challenge

Here's the code for the Eye Tracking Challenge in the paper "

> **Co-designing a Sub-millisecond Latency Event-based Eye Tracking System with Submanifold Sparse CNN**  
> Baoheng Zhang, Yizhao Gao, Jingyuan Li, Hayden Kowk-Hay So  
> (CVPR workshop 2024)
> 

The project mainly consists of three parts
- Software model training on event-based datasets with sparsity and quantization 
- Hardware design optimization (use constrained optimization to search for optimal mapping)
- Hardware synthesis, implementation and evaluation


## Installation of the environment/tools

### Vivado
This project depends on Vivado 2020.2. Please download and follow the installation guide from [xilinx](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html).
If you use newer vision, you might need to modify some project tcl by yourself.



### SCIPOPT for Hardware Optimization
To install SCIP Optimization Suite, please refer to [Installation guide](https://www.scipopt.org/doc/html/md_INSTALL.php). Note that SCIP must be compiled with `-DTPI=tny` to support concurrent solver.

After installation, please export your `$SCIPOPTDIR`
```bash
export SCIPOPTDIR=/path/to/scipoptdir
```



### Python

Assuming you have download the [anaconda](https://www.anaconda.com/download). You can create a new environment by:

```bash
conda create -n ESDA python=3.8
conda activate ESDA
```


Then install the required packages by:
```bash
pip3 install -r requirements.txt
```
**Important:** Make sure `$SCIPOPTDIR` is set up first before you install pysciopt.


For Pytorch installation, make sure you have installed the [nvidia driver](https://www.nvidia.com/download/index.aspx) successfully. Then you can refer to [Pytorch](https://pytorch.org/get-started/previous-versions/) to download the correct Pytorch 1.8 version.


For example, if your cuda version is 11.X, you can use
```bash
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

Finally you need to install the [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine):
```bash
cd software
conda install -c conda-forge cudatoolkit-dev
python setup.py install
```
(This is a temporal solution as we modified source code of Minkowski Engine for quantization. Will modify later.)


## Dataset 

The dataset is available at [link](https://www.kaggle.com/competitions/event-based-eye-tracking-ais2024/data). 
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

The checkpoints are available in the `weights` folder. 
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
For example, if you want to retrain MobileNetV2, assuming you are currently in the `software` folder, you can run the following command:

```bash
cd software
python main.py --config_file=configs/float32/MobileNetV2.json -s exp/float32/MobileNetV2
python main.py --config_file=configs/int8/MobileNetV2.json -s exp/int8/MobileNetV2 --shift_bit 16 --bias_bit 16 --load [path to float32 model]                  
```
, where the **[path to float32 model]** is the float32 model is saved in `exp/float32` folder you want to finetune. 
The final int8 model will be saved in `exp/int8` folder.

## Integer Inference

After obtaining int8 model, before hardware synthesis, you need to generate the integer model. Assuming you are currently in the `software` folder, you can run the following command:

```bash
python int_inference.py --config_file=configs/int8/MobileNetV2.json --checkpoint=[path to int8 model] --shift_bit 16 --bias_bit 16 --int_folder int/MobileNetV2
```
The **path to int8 model** is the quantized model trained in the previous step. The generated int8 model will be saved in `int/MobileNetV2` folder.

After the steps above, you can refer to [optimization](../optimization/README.md) to conduct the following steps.
