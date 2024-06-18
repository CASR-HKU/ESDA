# ESDA: A Composable Dynamic Sparse Dataflow Architecture for Efficient Event-based Vision Processing on FPGA

This repo contains the implementation for

> **ESDA: A Composable Dynamic Sparse Dataflow Architecture for Efficient Event-based Vision Processing on FPGA**  
> Yizhao Gao, Baoheng Zhang, Yuhao Ding, Hayden So  
> (FPGA 2024)

ESDA is a framework for building customized DNN accelerators for event-based vision tasks. It leverages the spatial sparsity of event-based input by a novel dynamic sparse dataflow architecture. This is achieved by formulating the computation of each dataflow module as a unified token-feature computation scheme. To enhance the spatial sparsity, ESDA also incorporates [Submanifold Sparse Convolution](https://arxiv.org/abs/1706.01307) to build our DNN models. 


The project mainly consists of three parts
- Software model training on event-based datasets with sparsity and quantization 
- Hardware design optimization (use constrained optimization to search for optimal mapping)
- Hardware synthesis, implementation and evaluation



If you are going to repoduce our **Artifact** on FPGA'24, Please refer to [evaluation.md](evaluation.md)

# *Latest News* 🔥

- [2024/06] We participate in the we also participate in the [Eye-tracking challenge](https://www.kaggle.com/competitions/event-based-eye-tracking-ais2024). 
The source code for the challenge is in `eye_tracking` branch. 
More details can be found in the 'eye_tracking' branch by 
```bash
git checkout eye_tracking
```

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

We use five datasets for the project: [README (1).md](..%2F..%2F..%2FDownloads%2FREADME%20%281%29.md)
[DvsGesture](https://research.ibm.com/interactive/dvsgesture/), 
[RoShamBo17](http://sensors.ini.uzh.ch/databases.html), 
[ASL-DVS](https://github.com/PIX2NVS/NVS2Graph), 
[N-MNIST](https://www.garrickorchard.com/datasets/n-mnist), and 
[N-Caltech101](https://www.garrickorchard.com/datasets/n-caltech101)

More about dataset preparation, please refer to software [readme](software/README.md).


## Overall Design Flow
The model training source code lies in `software` folder. After obtained a trained model, use toolflows in `optimization` folder to generate hardware configuration. Finally, use the hardware template and makefile in `hardware` folder to generate vitis_hls, vivado projects and synthesis your bitstream. 


## Eye-tracking challenge
Apart from the FPGA'24 artifact, 

## Acknowledgement
ESDA is inspired by and relies on many exisitng open-source libraries, including [Asynet](https://github.com/uzh-rpg/rpg_asynet), [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [HAWQ](https://github.com/Zhen-Dong/HAWQ), [AGNA](https://github.com/CASR-HKU/AGNA-FCCM2023), [DPCAS](https://github.com/CASR-HKU/DPACS) and etc. 


