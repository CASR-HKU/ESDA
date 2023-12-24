# Software model training

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