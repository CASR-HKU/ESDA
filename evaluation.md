

# Reproduce Results in FPGA'24 Paper


## Overall Comprehensive Software/Hardware Performance in Table 1

### Software model accuracy 


```bash
cd ~/ESDA/software
conda activate ESDA
```
We have prepared the model checkpoint in our server. Since github does not accept large file, we will archive our models in the final datahub with DOI. For now, you can directly run the following commands to generate the accuracy results for each dataset in **Table 1** in the paper.


```bash
cd software # Step back to the 'software' folder
python main.py --bias_bit 16 --settings_file=weights/ASL_w0p5/settings.yaml --load weights/ASL_w0p5/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/ASL_2929/settings.yaml --load weights/ASL_2929/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_slice_1890/settings.yaml --load weights/DVS_slice_1890/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_slice_w0p5/settings.yaml --load weights/DVS_slice_w0p5/ckpt.best.pth.tar --shift_bit 16 -e
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



## Hardware performance

We have already synthesised all the designs in **~/ESDA/eventNet/hw/** on our server. To evaluate the bitstreams directly, run the below commands.
There are 8 folders in the directory, each folder contains one hardware implementation. 


To begin with, make sure you are using the **esda** environment.

```bash
conda activate ESDA
```

### Hardware evalution: run our archived bitstreams

All the generated hardware/bitstreams are located in **~/ESDA/eventNet/hw/**. You can directly conduct performance evalution using the following commands.


####  Latency and power consumption evaluation

```bash
cd ~/ESDA/eventNet/hw/ASL_0p5_shift16-zcu102_80res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd ~/ESDA/eventNet/hw/ASL_2929_shift16-zcu102_80res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd ~/ESDA/eventNet/hw/DVS_1890_shift16-zcu102_80res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd ~/ESDA/eventNet/hw/DVS_w0p5_shift16-zcu102_60res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd ~/ESDA/eventNet/hw/NMNIST_shift16-zcu102_60res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd ~/ESDA/eventNet/hw/Roshambo_shift16-zcu102_60res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd ~/ESDA/eventNet/hw/NCal_2751_shift32-zcu102_80res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

```bash
cd ~/ESDA/eventNet/hw/NCal_w0p5_shift32-zcu102_50res/full/
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

The latency will be displayed in the terminal respectively.

After finish the commands above, you can extract the overall results by

```bash
cd ~/ESDA/hardware
conda activate base # We need python >=3.9 here 
python baseline_extract.py ../eventNet/hw/ --extract_large
```
The overall results will be saved in **../eventNet/hw/extract_large.csv**, which will match the performance result of Table 1.


#### End-to-end result verification 

To verify the end-to-end inference results, run the following commands.
The **python sw_e2e.py** script will generate the software end-to-end inference results for one test input, while the **make e2e_inference** will obtain the hardware end-to-end inference result on the same input.

```bash
cd ~/ESDA/eventNet/hw/ASL_0p5_shift16-zcu102_80res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd ~/ESDA/eventNet/hw/ASL_2929_shift16-zcu102_80res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd ~/ESDA/eventNet/hw/DVS_1890_shift16-zcu102_80res/full/
python sw_e2e.py
make e2e_inference
```

```bash 
cd ~/ESDA/eventNet/hw/DVS_w0p5_shift16-zcu102_60res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd ~/ESDA/eventNet/hw/NMNIST_shift16-zcu102_60res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd ~/ESDA/eventNet/hw/Roshambo_shift16-zcu102_60res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd ~/ESDA/eventNet/hw/NCal_2751_shift32-zcu102_80res/full/
python sw_e2e.py
make e2e_inference
```

```bash
cd ~/ESDA/eventNet/hw/NCal_w0p5_shift32-zcu102_50res/full/
python sw_e2e.py
make e2e_inference
```


### Resynthesis your hardware (optional)
If you want to resysthesis the bitstream from scratch by yourself, here shows the example instructions for `DVS_1890_shift16-zcu102_80res`. (This can take around 12 hours for one design.)

```bash
cd ~/EDSA/hardware
mkdir MyPrj && cd MyPrj # create a new folder for your projects
cp -r ../template_e2e DVS_1890_shift16-zcu102_80res # For roshambo, use '../template_e2e_roshambo'
cp ../cfgs/DVS_1890_shift16-zcu102_80res.json DVS_1890_shift16-zcu102_80res/cfg.json
cd DVS_1890_shift16-zcu102_80res

conda activate ESDA 
make gen  # Generate hls code
make ip_all  # Vitis systhesis
make hw_all  # Vivado systhesis
make make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'" # Evaluate the latency and power consumption
make e2e_inference  # Conduct e2e inference on board
python sw_e2e.py  # Conduct e2e inference with Pytorch
```



## Benchmark co-simulation in Figure 13

If you want to reproduce the results compared with dense architecture in Figure 13, run the following instructions (this can takes hours to finish):


```bash
cd ~/ESDA/hardware/benchmark_results
make all -j4
#after finish all co-sim
python benchmark_extract.py
```

You can see: 
```
csv saved to benchmark_results/DVS_mobilenet_0707_0p5-zcu102_50res_new.csv
npy saved to benchmark_results/DVS_mobilenet_0707_0p5-zcu102_50res_new.npy
```

The benchmark results are then stored in **benchmark_results/DVS_mobilenet_0707_0p5-zcu102_50res_new.csv**