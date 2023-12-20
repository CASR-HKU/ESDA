

## Introduction

Here is the documents for the artifact evaluation.


## Procedures 

### 1. Software evaluation

After logging in our server, go to the directory '/vol/datastore/EDSA/ESDA/software/' by 

```bash
cd /vol/datastore/aeguest/ESDA/
```

For convenience, we has create a conda environment for the AE. 
You can activate the environment by

```bash
conda activate esda
```

Due to the large size of dataset, we have prepared the dataset in our server in **/vol/datastore/event_dataset/**. You can directly run the following commands to generate the accuracy results for each dataset in **Table 1** in the paper.

```bash
python main.py --bias_bit 16 --settings_file=weights/ASL_w0p5/settings.yaml --load weights/ASL_w0p5/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/ASL_2929/settings.yaml --load weights/ASL_2929/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_1890/settings.yaml --load weights/DVS_1890/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_w0p5/settings.yaml --load weights/DVS_w0p5/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/NMNIST/settings.yaml --load weights/NMNIST/ckpt.best.pth.tar --shift_bit 16 -e 
python main.py --bias_bit 16 --settings_file=weights/Roshambo/settings.yaml --load weights/Roshambo/ckpt.best.pth.tar --shift_bit 16 -e 
python main.py --bias_bit 32 --settings_file=weights/NCal_2751/settings.yaml --load weights/NCal_2751/ckpt.best.pth.tar --shift_bit 32 -e
python main.py --bias_bit 32 --settings_file=weights/NCal_w0p5/settings.yaml --load weights/NCal_w0p5/ckpt.best.pth.tar --shift_bit 32 -e
```

The results will be displayed in the terminal respectively.
For example, if you are evaluating the **NMNIST** dataset, the result displayed should be:

```bash
Total sample num is 19942
Number of ALL parameters: 281184
Number of parameters except BN: 270560
Valid: -1 | loss: 0.0207 | Top-1: 99.00 | Top-5: 99.98: 100%####################################################################| 78/78 [00:12<00:00,  6.43it/s]
* Epoch -1 - Prec@1 99.002 - Prec@5 99.985 - Loss 0.032
99.0 99.98 0.03 281184 270560
```
where the Prec@1 is the accuracy result shown in the **Acc%** column in the table.


### 2. Hardware evaluation


We have already synthesis all the designs in folder in our server. To evaluate the bitstreams directly, you can go to **/vol/datastore/ESDA/aeguest/eventHW/** to run the [commands](#bitstreams-evaluation).
 There are 8 folders in the directory, each folder contains one hardware implementation. 

If you want to resynthesis the designs, please refer to [resynthesis](#resysthesis)

To begin with, make sure you are using the **esda** environment.

```bash
conda activate esda
```

#### Bitstreams evaluation

All the generated hardware are prepared in **/vol/datastore/ESDA/aeguest/eventHW/**. You can directly conduct evalution using the following commands.


1. Latency and power consumption evaluation

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/ASL_0p5_shift16-zcu102_80res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/ASL_2929_shift16-zcu102_80res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/DVS_1890_shift16-zcu102_80res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/DVS_w0p5_shift16-zcu102_60res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/NMNIST_shift16-zcu102_60res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/Roshambo_shift16-zcu102_60res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/NCal_2751_shift32-zcu102_80res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/NCal_w0p5_shift32-zcu102_50res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

The latency will be displayed in the terminal respectively, while the power consumption will be saved in the csv file "".



2. End-to-end evaluation

To evaluate the end-to-end inference results, run the following commands.
The **python sw_e2e.py** script will generate the software end-to-end inference results, while the **make e2e_inference** hardware end-to-end inference results respectively.

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/ASL_0p5_shift16-zcu102_80res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/ASL_2929_shift16-zcu102_80res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/DVS_1890_shift16-zcu102_80res/full/
python sw_e2e.py
make e2e_inference
```

```bash 
cd /vol/datastore/ESDA/aeguest/eventHW/DVS_w0p5_shift16-zcu102_60res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/NMNIST_shift16-zcu102_60res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/Roshambo_shift16-zcu102_60res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/NCal_2751_shift32-zcu102_80res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd /vol/datastore/ESDA/aeguest/eventHW/NCal_w0p5_shift32-zcu102_50res/full/
python sw_e2e.py
make e2e_inference
```


#### Resysthesis
if you want to resysthesis the whole project, we have prepare the template for each implementation in **/vol/datastore/ESDA/aeguest/eventHW_tpl**.

```bash
mkdir resysthesis && cd resysthesis
cp -r /vol/datastore/ESDA/aeguest/eventHW_tpl/ .
```

The project structure will be generated like this:

```
├── resysthesis
│   ├── DVS_1890_shift16-zcu102_80res
│   │   ├── full
│   │   ├── Makefile
│   ├── DVS_0p5_shift16-zcu102_60res
│   ├── ASL_0p5_shift16-zcu102_80res
│   ├── ASL_2929_shift16-zcu102_80res
│   ├── NCal_2751_shift32-zcu102_80res
│   ├── NCal_w0p5_shift32-zcu102_50res
│   ├── NMNIST_shift16-zcu102_60res
│   ├── Roshambo_shift16-zcu102_80res
```

Then you can enter one of the folder to conduct overall systhesis.

For example, if you want to resysthesis the group DVS_1890_shift16-zcu102_80res, 

```bash
# Assuming you are in the folder 'resysthesis'
cd DVS_1890_shift16-zcu102_80res/full
make gen  # Generate sample
make ip_all  # Vitis systhesis
make hw_all  # Vivado systhesis
make make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'" # Evaluate the latency and power consumption
make e2e_inference  # Conduct e2e inference on board
python sw_e2e.py 
```
