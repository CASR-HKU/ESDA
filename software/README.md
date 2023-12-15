Installations

Python
Minkowski Engine
Pytorch


Evaluate trained model

1. Download the dataset from 

Follow the dataset preparation steps from ....
You can skip the step while evaluating in a1

2. Download pretrained model from and put them to weights folder
You can skip the step while evaluating in a1

3. Run the following command to evaluate the model

python main.py --bias_bit 16 --settings_file=weights/ASL_w0p5/settings.yaml --load weights/ASL_w0p5/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/ASL_2929/settings.yaml --load weights/ASL_2929/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_1890/settings.yaml --load weights/DVS_1890/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_w0p25/settings.yaml --load weights/DVS_w0p25/ckpt.best.pth.tar --shift_bit 16 -e
python main.py --bias_bit 16 --settings_file=weights/NMNIST/settings.yaml --load weights/NMNIST/ckpt.best.pth.tar --shift_bit 16 -e 
python main.py --bias_bit 16 --settings_file=weights/Roshambo/settings.yaml --load weights/Roshambo/ckpt.best.pth.tar --shift_bit 16 -e 
python main.py --bias_bit 16 --settings_file=weights/NCal_2751/settings.yaml --load weights/NCal_2751/ckpt.best.pth.tar --shift_bit 32 -e
python main.py --bias_bit 16 --settings_file=weights/NCal_w0p5/settings.yaml --load weights/NCal_w0p5/ckpt.best.pth.tar --shift_bit 32 -e
python main.py --bias_bit 16 --settings_file=weights/DVS_w0p5/settings.yaml --load weights/DVS_w0p5/ckpt.best.pth.tar --shift_bit 16 -e 

It will generate the accuracy result:




Generate 



Train the model

1. Training float32 model using the following commands

python main.py --settings_file=config/config_AE/float32/ASL_2929.yaml -s exp_float32/ASL_2929
python main.py --settings_file=config/config_AE/float32/ASL_w0p5.yaml -s exp_float32/ASL_w0p5
python main.py --settings_file=config/config_AE/float32/DVS_1890.yaml -s exp_float32/DVS_1890
python main.py --settings_file=config/config_AE/float32/DVS_w0p5.yaml -s exp_float32/DVS_w0p5
python main.py --settings_file=config/config_AE/float32/NMNIST.yaml -s exp_float32/NMNIST
python main.py --settings_file=config/config_AE/float32/Roshambo.yaml -s exp_float32/Roshambo
python main.py --settings_file=config/config_AE/float32/NCal_2751.yaml -s exp_float32/NCal_2751
python main.py --settings_file=config/config_AE/float32/NCal_w0p5.yaml -s exp_float32/NCal_w0p5

The training results will be stored in the -s folder respectively.


2. Training int8 model

After obtaining the float32 model, we can train the int8 model using the following commands.

python main.py --settings_file=config/config_AE/int8/ASL_2929.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/ASL_2929 --load exp_float32/ASL_2929/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/ASL_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/ASL_w0p5 --load exp_float32/ASL_w0p5/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/DVS_1890.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/DVS_1890 --load exp_float32/DVS_1890/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/DVS_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/DVS_w0p5 --load exp_float32/DVS_w0p5/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/NMNIST.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NMNIST --load exp_float32/NMNIST/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/Roshambo.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/Roshambo --load exp_float32/Roshambo/ckpt.best.pth.tar --shift_bit 16
python main.py --settings_file=config/config_AE/int8/NCal_2751.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NCal_2751 --load exp_float32/NCal_2751/ckpt.best.pth.tar --shift_bit 32
python main.py --settings_file=config/config_AE/int8/NCal_w0p5.yaml --epochs 100 --fixBN_ratio 0.3 -s exp_int8/NCal_w0p5 --load exp_float32/NCal_w0p5/ckpt.best.pth.tar --shift_bit 32


Software model searching

You can generate the model configurations by running the following commands:

python search_sw.py -n 100

It will generate 100 model configurations in terminal

