

# Reproduce Results in FPGA'24 Paper

Here is the documents for the reproduce the results in the paper.

## Overall Comprehensive Software/Hardware Performance in Table 1

### Model accuracy 


```bash
cd ~/ESDA/software
conda activate esda
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



### Hardware Performance

We have already synthesis all the designs in folder in our server. To evaluate the bitstreams directly, you can go to **~/ESDA/eventNet/hw/** to run the [commands](#bitstreams-evaluation).
 There are 8 folders in the directory, each folder contains one hardware implementation. 

If you want to resynthesis the designs, please refer to [resynthesis](#resysthesis)

To begin with, make sure you are using the **esda** environment.

```bash
conda activate esda
```

### Hardware: run with our archived Bitstreams

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

After finish the commands above, you can generate the overall results by



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
If you want to resysthesis the bitstream from scratch by yourself, you can follow the below example instructions for `DVS_1890_shift16-zcu102_80res`. (This can take up to 12 hours to resynthesis one design.)

```bash
cd ~/EDSA/hardware
mkdir MyPrj && cd MyPrj
cp -r ../template_e2e DVS_1890_shift16-zcu102_80res # For roshambo, use '../template_e2e_roshambo'
cp ../cfgs/DVS_1890_shift16-zcu102_80res.json DVS_1890_shift16-zcu102_80res/cfg.json
cd DVS_1890_shift16-zcu102_80res

conda activate ESDA 
make gen  # Generate sample
make ip_all  # Vitis systhesis
make hw_all  # Vivado systhesis
make make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'" # Evaluate the latency and power consumption
make e2e_inference  # Conduct e2e inference on board
python sw_e2e.py  # Conduct e2e inference with Pytorch
```



## Benchmark simulation in Figure 13

If you want to reproduce the results compared with dense architecture in Figure 13, run the following instructions:


```bash
cd ~/ESDA/hardware/benchmark_results
make all -j4
python benchmark_extract.py
```

You can see: 
```
csv saved to benchmark_results/DVS_mobilenet_0707_0p5-zcu102_50res_new.csv
npy saved to benchmark_results/DVS_mobilenet_0707_0p5-zcu102_50res_new.npy
```

Where the benchmark results are stored in **benchmark_results/DVS_mobilenet_0707_0p5-zcu102_50res_new.csv**