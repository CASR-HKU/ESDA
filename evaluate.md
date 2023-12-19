

## Introduction

Here is the script for artifact evaluation.....


## Procedures 

### 1. Software evaluation

After logging in a1, go to the directory '/vol/datastore/EDSA/ESDA/software/' by 

```bash
cd /vol/datastore/EDSA/ESDA/software/
```

Then activate the environment by

```bash
conda activate esda
```

Finally run the following commands to generate the results.
```bash
python main.py --bias_bit 16 --settings_file=weights/ASL_w0p5/settings.yaml --load weights/ASL_w0p5/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/ASL_2929/settings.yaml --load weights/ASL_2929/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_1890/settings.yaml --load weights/DVS_1890/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_w0p5/settings.yaml --load weights/DVS_w0p5/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/NMNIST/settings.yaml --load weights/NMNIST/ckpt.best.pth.tar --shift_bit 16 -e 
python main.py --bias_bit 16 --settings_file=weights/Roshambo/settings.yaml --load weights/Roshambo/ckpt.best.pth.tar --shift_bit 16 -e 
python main.py --bias_bit 16 --settings_file=weights/NCal_2751/settings.yaml --load weights/NCal_2751/ckpt.best.pth.tar --shift_bit 32 -e
python main.py --bias_bit 16 --settings_file=weights/NCal_w0p5/settings.yaml --load weights/NCal_w0p5/ckpt.best.pth.tar --shift_bit 32 -e
```

The results will be displayed in the terminal respectively.


### 2. Int8 model inference







### 3. Hardware evaluation

After logging in a1, go to the directory '/vol/datastore/EDSA/AE_test/eventHW/' by 

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/
```

Then activate the environment by

```bash
conda activate esda
```


There are 8 folders in the directory, each folder contains one hardware implementation. 

#### 1. Latency and power consumption evaluation

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/ASL_0p5_shift16-zcu102_80res
make evalaute_hw EVAL_TARGET="e2e ARG_NUM_RUN=-1" --enable_pm
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/ASL_2929_shift16-zcu102_80res
make evalaute_hw EVAL_TARGET="e2e ARG_NUM_RUN=-1" --enable_pm
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/DVS_1890_shift16-zcu102_80res
make evalaute_hw EVAL_TARGET="e2e ARG_NUM_RUN=-1" --enable_pm
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/DVS_w0p5_shift16-zcu102_60res
make evalaute_hw EVAL_TARGET="e2e ARG_NUM_RUN=-1" --enable_pm
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/NMNIST_shift16-zcu102_60res
make evalaute_hw EVAL_TARGET="e2e ARG_NUM_RUN=-1" --enable_pm
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/Roshambo_shift16-zcu102_60res
make evalaute_hw EVAL_TARGET="e2e ARG_NUM_RUN=-1" --enable_pm
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/NCal_2751_shift32-zcu102_80res
make evalaute_hw EVAL_TARGET="e2e ARG_NUM_RUN=-1" --enable_pm
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/NCal_w0p5_shift32-zcu102_50res
make evalaute_hw EVAL_TARGET="e2e ARG_NUM_RUN=-1" --enable_pm
```

The latency will be displayed in the terminal respectively, while the power consumption will be saved in the csv file "".


#### 2. End-to-end evaluation

To evaluate the end-to-end inference results, run the following commands.
The **python** script will generate the software end-to-end inference results, while the **make** hardware end-to-end inference results respectively.

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/ASL_0p5_shift16-zcu102_80res
python sw_e2e.py
make evalaute_e2e
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/ASL_2929_shift16-zcu102_80res
python sw_e2e.py
make evalaute_e2e
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/DVS_1890_shift16-zcu102_80res
python sw_e2e.py
make evalaute_e2e
```

```bash 
cd /vol/datastore/EDSA/AE_test/eventHW/DVS_w0p5_shift16-zcu102_60res
python sw_e2e.py
make evalaute_e2e
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/NMNIST_shift16-zcu102_60res
python sw_e2e.py
make evalaute_e2e
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/Roshambo_shift16-zcu102_60res
python sw_e2e.py
make evalaute_e2e
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/NCal_2751_shift32-zcu102_80res
python sw_e2e.py
make evalaute_e2e
```

```bash
cd /vol/datastore/EDSA/AE_test/eventHW/NCal_w0p5_shift32-zcu102_50res
python sw_e2e.py
make evalaute_e2e
```
