
# Hardware Synthesis and Implementation

## Prepare your FPGA board
We use ZCU102 board with [PYNQ](http://www.pynq.io/board.html) overlay in our design, and some commands and tcl scripts are hardend for our board. If you have a different board and setting, please use Vitis, Vitis_HLS, or Vivado to regenerate your project tcl. 



## Obtaining Project 

```bash
cd hardware
python gen_prj.py gen_full --cfg_name <cfg-name> --cfg_path <path-to-DES-folder> --tpl_dir template_e2e --dst_path <path-to-destination>
```

For example, after running the codes in the optimization folder, you will get the following file structure:

```
EDSA
├── software
├── hardware
├── optimization
├── eventNet
│   ├── model
│   ├── DSE
│   │   ├── MobileNetV2
│   │   │   ├── en-config.json
│   │   │   ├── en-gpkit.model
│   │   │   ├── en-gpkit.sol
│   │   │   ├── en-result.json
│   │   │   ├── en-scip.cip
│   │   │   ├── en-scip.log
│   │   │   ├── en-scip.sol
│   │   │   ├── main.log
```

Build the hardware project by using:

```bash
# Make sure you are in the root directory

cd hardware
python gen_prj.py gen_full --cfg_name MobileNetV2 --cfg_path ../eventNet/DSE --tpl_dir template_e2e --dst_path ../eventNet/HW
```
The hardware project will be saved in the **eventNet/HW/** folder.



## HLS and Vivado Synthesis

The complete hardware generation and evaluation process can be divided into 4 steps 
1. Generating HLS source code
2. Run vitis_hls and generate ip
3. Run vivado and extract hardware
4. Evaluation performance on the board


The current tcl script mainly target our ZCU102 board. Besides, our board is running pynq overlay. 


###  Generating HLS source code

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
│   │   ├── MobileNetV2
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

```bash
# Make sure you are in the root directory
cd eventNet/HW/MobileNetV2/full
make gen
```

The data txt will be generated in the **data** folder inside the project root.



### Run vitis_hls and generate ip

The next step is to run vitis and generate ip. To achieve it, entering the project folder and run make ip_all by:

```bash
cd <Path-to-the-generated-project-folder>
make ip_all
```

The results and logs will be stored in the **prj** folder inside the hardware project root. For example, 
to extract the ip of **MobileNetV2** model, you should:

```bash
# Make sure you are in the root directory
cd eventNet/HW/MobileNetV2/full
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
You can find the output bistream inside the **eventNet/HW/DVS_1890_shift16-zcu102_80res/full/hw** folder.




## Hardware performance evaluation

The below command will connect our ZCU102 server and run performance evaluation. 
If you have a different board or setting, you can refer to `board` folder to see how we run design. 

#### 1. Evaluate latency and power

```bash
# Make sure you are in the root directory
cd eventNet/HW/MobileNetV2/full
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
cd eventNet/HW/MobileNetV2/full
make e2e_inference
```

(Or let them directly run on the board?)
```bash
# Make sure you are in the root directory
cd $ESDA_HOME/hardware/board
python3 hw_e2e.py 1 -d hw/DVS_1890/
```
