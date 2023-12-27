# Software model training

## Dataset Preparation

#### DVSGesture, ASLDVS and N-MNIST
To download the DVSGesture, ASLDVS and N-MNIST dataset, use the scripts **software/dataset/preprocess/get_dataset.py**
```bash
# Make sure you are in the software folder
python dataset/preprocess/get_dataset.py DVSGesture -p data
python dataset/preprocess/get_dataset.py NMNIST -p data
python dataset/preprocess/get_dataset.py ASLDVS -p data
```

#### Roshambo
For RoShamBo17 dataset download, go to [Roshambo17](https://docs.google.com/document/d/e/2PACX-1vTNWYgwyhrutBu5GpUSLXC4xSHzBbcZreoj0ljE837m9Uk5FjYymdviBJ5rz-f2R96RHrGfiroHZRoH/pub#h.uzavf0ex4d2e) and save the file in **software/data/Roshambo** path.


#### N-Caltech101
For N-Caltech101 dataset, we followed the process in [asynet](https://github.com/uzh-rpg/rpg_asynet) by using the following scripts. You can use tonic too.

```bash
cd data
wget http://rpg.ifi.uzh.ch/datasets/gehrig_et_al_iccv19/N-Caltech101.zip
unzip N-Caltech101.zip
rm N-Caltech101.zip
cd ..
```

After downloading all the datasets, your file structure will be:
```
EDSA
├── hardware
├── optimization
├── software
│   ├── data
│   │   ├── ASLDVS
│   │   ├── DVSGesture
│   │   ├── N-Caltech101
│   │   ├── NMNIST
│   │   ├── Roshambo/dataset_nullhop
```


## Dataset preprocess

Before feed events into your DNN model, there are different ways to can convert events into 2D/frame-like representation. In this work, we mainly use histogram for in a fixed time interval. For some dataset, we also perform simple denoise of event to increase input sparsity.  
Below shows some of our preprocess scripts.


#### NMNIST & NCaltech101

```bash
# Make sure you are in the 'software' folder.
# For NMNIST
python data_preprocess.py --settings_file=config/preprocess/NMNIST_settings_sgd.yaml --preprocess -s dataset/preprocess/NMNIST_preprocessed --window_size 200000 --overlap_ratio 0.5
# For NCaltech101
python data_preprocess.py --settings_file=config/preprocess/NCAL_settings_sgd.yaml --preprocess -s dataset/preprocess/NCal_preprocessed --window_size 0.1 --overlap_ratio 0.5
```

#### DVSGesture

To preprocess DVSGesture dataset, use the **dvs_preprocess.py** in **software/dataset/preprocess** folder:
```bash
# Make sure you are in the 'software' folder.
cd dataset/preprocess
python dvs_preprocess.py --input_dir ../../DvsGesture --output_dir ../../dvs_gesture_clip
```



Once you finished, the file structure should be like this:

```
EDSA
├── software
├── hardware
├── optimization
├── data
│   ├── ASLDVS
│   ├── DvsGesture
│   ├── dvs_gesture_clip
│   ├── NCal
│   ├── NCal_processed
│   ├── NMNIST
│   ├── NMNIST_processed
│   ├── Roshambo/dataset_nullhop
```




## Training float32 model

```bash
cd software
python main.py --settings_file=<path-to-float32-config-file> -s <path-to-result-folder>
```
And the model will be saved in your ```<path-to-result-folder>```


For example, after downloading the dataset, you can conduct training float32 model using the following commands

```bash
python main.py --settings_file=config/default/float32/ASL_w0p5.yaml -s exp_float32/ASL_w0p5  # For ASL-DVS
python main.py --settings_file=config/default/float32/DVS_w0p5.yaml -s exp_float32/DVS_1890  # For DvsGesture
python main.py --settings_file=config/default/float32/NMNIST.yaml -s exp_float32/NMNIST   # For N-MNIST
python main.py --settings_file=config/default/float32/Roshambo.yaml -s exp_float32/Roshambo  # For RoShamBo17
python main.py --settings_file=config/default/float32/NCal_w0p5.yaml -s exp_float32/NCal_w0p5   # For N-Caltech101
```

## Training int-8 quantized model

After training with float32 model, you can train the int8 model using the following commands.
```bash
python main.py --settings_file=<path-to-float32-config-file> -s <path-to-result-folder> --load <path-to-float32-model> --shift_bit <shift-bit> --fixBN_ratio <fixBN-ratio>
```
And the model will be saved in your ```<path-to-result-folder>```.

For example, assuming you have trained the float32 model above, you can train the int8 model using the following commands:

```bash
python main.py --settings_file=config/default/int8/DVS_1890.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/DVS_1890 --load exp_float32/DVS_1890/ckpt.best.pth.tar --shift_bit 16
```


### 3. Export an Int8 model

After obtaining the trained int8 model, before conducting hardware systhesis, use **int_inference.py** to generate necessary software configuration and input data.

```bash
python int_inference.py --settings_file=<path-to-config-file> --load <path-to-trained-model-path> --bias_bit <bias-bit> --shift_bit <scale-bit> -e --int_dir <path-to-model-folder>
```


```bash
python int_inference.py --bias_bit 16 --settings_file=exp_int8/DVS_1890/settings.yaml --load exp_int8/DVS_1890/ckpt.best.pth.tar --shift_bit 16 -e --int_dir ../eventNet/model/DVS_1890_shift16
```

```
EDSA
├── software
│   ├── data
│   │   ├── ASLDVS
│   │   ├── dvs_gesture_clip
│   │   ├── NCal
│   │   ├── NCal_processed
│   │   ├── NMNIST
│   │   ├── NMNIST_processed
│   │   ├── Roshambo/dataset_nullhop
├── hardware
├── optimization
├── eventNet
│   ├── model
│   │   ├── DVS_1890_shift16
```

After the steps above, you can refer to [optimization](../optimization/README.md) to conduct the following steps.