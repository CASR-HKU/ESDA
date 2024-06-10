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
