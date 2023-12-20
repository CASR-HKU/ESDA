

This is the code for the paper "xxx"

The whole pipeline contains three parts
1. Software model training and evaluation
2. Hardware configuration optimization
3. Hardware systhesis
4. Hardware performance evaluation


If you are going to conduct **Artifact evaluation**, Please refer to [evaluate](evaluate.md)


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
cd software
python setup.py install
```




## Software model training and evaluation

### 1. Dataset preparation

Please refer to [dataset preparation](doc/dataset.md) for the details.

Once you finished, you will obtain the data structure like this

```
EDSA
├── software
├── hardware
├── optimization
├── data
│   ├── ASLDVS
│   ├── dvs_gesture_clip
│   ├── NCal
│   ├── NMNIST
│   ├── Roshambo
```


### 2. Software model training

All the training progress are done in the **software** folder. 

(1) Training float32 model

```bash
python main.py --settings_file=<path-to-float32-config-file> -s <path-to-result-folder>
```
And the model will be saved in your <path-to-result-folder>


For example, after downloading the dataset, you can conduct training float32 model using the following commands

```bash
python main.py --settings_file=config/default/float32/ASL_w0p5.yaml -s exp_float32/ASL_w0p5  # For ASL-DVS
python main.py --settings_file=config/default/float32/DVS_w0p5.yaml -s exp_float32/DVS_1890  # For DvsGesture
python main.py --settings_file=config/default/float32/NMNIST.yaml -s exp_float32/NMNIST   # For N-MNIST
python main.py --settings_file=config/default/float32/Roshambo.yaml -s exp_float32/Roshambo  # For RoShamBo17
python main.py --settings_file=config/default/float32/NCal_w0p5.yaml -s exp_float32/NCal_w0p5   # For N-Caltech101
```

To simplify, we will only demonstrate the project **DVS_1890** as the exmaple for introduction.


(2) Training int8 model

After training with float32 model, you can train the int8 model using the following commands.
```bash
python main.py --settings_file=<path-to-float32-config-file> -s <path-to-result-folder> --load <path-to-float32-model> --shift_bit <shift-bit> --fixBN_ratio <fixBN-ratio>
```
And the model will be saved in your <path to result folder>.

For example, assuming you have trained the float32 model above, you can train the int8 model using the following commands:

```bash
python main.py --settings_file=config/default/int8/DVS_1890.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/DVS_1890 --load exp_float32/DVS_1890/ckpt.best.pth.tar --shift_bit 16
```


### 3. Int8 model inference






## Hardware configuration optimization

### 1. Hardware searching

After obtaining the int8 model, the next step is to optimize the hardware configuration based on your model architecture and the dataset.


```bash
# Assuming you are in the root directory
cd optimization
python eventnet.py --model_path <path-of-models-root-folder> --eventNet_path <path-of-hw-config-folder> --model_name  <name-of-the-target-folder> --eventNet_name <name-of-the-target-hw-config> --results_path <path-of-stored-result>
```

For example, assuming you have generate the corresponding model structures and input data using the commands above.
You will obtain a folder of model structure and feature files

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

You can conduct hardware configuration optimization by the following commands
```bash
# Make sure you are in the root directory
python eventnet.py --model_path ../eventNet/model --eventNet_path /vol/datastore/EDSA/eventNeteventNetConfig --model_name DVS_1890_shift16 --eventNet_name zcu102_80res --results_path ../eventNet/DSE
```


### 2. Project generation


```bash
# Make sure you are in the root directory
cd hardware
# For roshambo
python gen_prj.py gen_full --cfg_name <cfg-name> --cfg_path <path-to-DES-folder> --tpl_dir template_e2e_roshambo --dst_path <path-to-destination>
# For other datasets
python gen_prj.py gen_full --cfg_name <cfg-name> --cfg_path <path-to-DES-folder> --tpl_dir template_e2e --dst_path <path-to-destination>
```

For example, assuming you generate the DSE result using the commands above, you will generate the config files in the eventNet/DSE file

```
EDSA
├── software
├── hardware
├── optimization
├── eventNet
│   ├── model
│   ├── DSE
│   │   ├── ASL_0p5_shift16-zcu102_80res
│   │   │   ├── en-config.json
│   │   │   ├── en-gpkit.model
│   │   │   ├── en-gpkit.sol
│   │   │   ├── en-result.json
│   │   │   ├── en-scip.cip
│   │   │   ├── en-scip.log
│   │   │   ├── en-scip.sol
│   │   │   ├── main.log
│   │   ├── ASL_2929_shift16-zcu102_80res
│   │   ├── NMNIST_shift16-zcu102_60res
│   │   ├── Roshambo_shift16-zcu102_80res
│   │   ├── DVS_1890_shift16-zcu102_80res
│   │   ├── DVS_0p5_shift16-zcu102_60res
│   │   ├── NCal_w0p5_shift32-zcu102_50res
│   │   ├── NCal_2751_shift32-zcu102_80res
```

Then you can generate the corresponding hardware project by:

```bash
# Make sure you are in the root directory
cd hardware
python gen_prj.py gen_full --cfg_name ASL_0p5_shift16-zcu102_80res --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW
python gen_prj.py gen_full --cfg_name ASL_2929_shift16-zcu102_80res --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW
python gen_prj.py gen_full --cfg_name NMNIST_shift16-zcu102_60res --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW
python gen_prj.py gen_full --cfg_name Roshambo_shift16-zcu102_80res --cfg_path ../eventNet/DSE --tpl_dir template_e2e_roshambo --dst_path ../eventNet/HW
python gen_prj.py gen_full --cfg_name DVS_1890_shift16-zcu102_80res --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW
python gen_prj.py gen_full --cfg_name DVS_0p5_shift16-zcu102_60res --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW
python gen_prj.py gen_full --cfg_name NCal_w0p5_shift32-zcu102_50res --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW
python gen_prj.py gen_full --cfg_name NCal_2751_shift32-zcu102_80res --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW
```
The hardware project will be saved in the **eventNet/HW/** folder.



## Hardware systhesis

The complete hardware generation and evaluation process can be divided into 4 steps 
1. Generating samples
2. Run vitis and generate ip
3. Run vivado and extract hardware
4. Evaluation in the board


### 1. Generating samples

After generating the hardware projects, the first step generate the integer samples. You enter the generated project folder and use the following commands:

```bash
cd <Path-to-the-generated-project-folder>
make gen
```

For example, assuming you generated the hardware project above, you will see the following file structure.

```
EDSA
├── software
├── hardware
├── optimization
├── eventNet
│   ├── model
│   ├── DSE
│   ├── HW
│   │   ├── DVS_1890_shift16-zcu102_80res
│   │   │   ├── full
│   │   │   │   ├── prj
│   │   │   │   │   ├── hls.tcl
│   │   │   │   │   ├── vivado.tcl
│   │   │   │   │   ├── ...... (Other files)
│   │   │   │   ├── cfg.json
│   │   │   │   ├── gen_data.py
│   │   │   │   ├── gen_code.py
│   │   │   │   ├── linebuffer.h
│   │   │   │   ├── ...... (Other files)
│   │   │   ├── Makefile
```

Then you can enter each project and generate the integer samples by:

```bash
# Make sure you are in the root directory
cd eventNet/HW/DVS_1890_shift16-zcu102_80res/full
make gen
```

The data txt will be generated in the **data** folder inside the project root.



### 2. Run vitis and generate ip

The next step is to run vitis and generate ip. To achieve it, entering the project folder and run make ip_all by:

```bash
cd <Path-to-the-generated-project-folder>
make ip_all
```

The results and logs will be stored in the **prj** folder inside the hardware project root. For example, to extract the ip of **DVS_1890** model, you should:

```bash
# Make sure you are in the root directory
cd eventNet/HW/DVS_1890_shift16-zcu102_80res/full
make ip_all
```

You can check the results inside the **eventNet/HW/DVS_1890_shift16-zcu102_80res/full/prj** folder.


### 3. Run vivado and extract hardware

After extracting ip, you can use the following commands to generate the hardware:

```bash
cd <Path-to-the-generated-project-folder>
make hw_all
```

The hardware results will be stored in the **hw** folder. For example, to generate the ip of **DVS_1890** model, you should:

```bash
# Make sure you are in the root directory
cd eventNet/HW/DVS_1890_shift16-zcu102_80res/full
make hw_all
```
You can check the results inside the **eventNet/HW/DVS_1890_shift16-zcu102_80res/full/hw** folder.




## Hardware performance evaluation

Finally 

#### 1. Evaluate latency and power

```bash
# Make sure you are in the root directory
cd eventNet/HW/DVS_1890_shift16-zcu102_80res/full
make evaluate_hw EVAL_TARGET="e2e ARG_NUM_RUN='-1 --enable_pm'"
```

(Or let them directly run on the board?)
```bash
# Make sure you are in the root directory
cd $ESDA_HOME/hardware/board
python3 evaluate.py -1 -d hw/DVS_1890/
```

#### 2. End-to-end evaluation

```bash
# Make sure you are in the root directory
cd eventNet/HW/DVS_1890_shift16-zcu102_80res/full
make e2e_inference
```

(Or let them directly run on the board?)
```bash
# Make sure you are in the root directory
cd $ESDA_HOME/hardware/board
python3 hw_e2e.py 1 -d hw/DVS_1890/
```
