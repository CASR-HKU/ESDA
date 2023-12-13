

cmds = [
    "python main.py -s sparsity/DVS/base --settings_file=config_sparsity/DVS_base.yaml -e ",
    "python main.py -s sparsity/DVS/mink --settings_file=config_sparsity/DVS_mink.yaml -e ",
    "python main.py -s sparsity/DVS/mink_drop20 --settings_file=config_sparsity/DVS_mink_drop.yaml -e ",

    "python main.py -s sparsity/DVS/base --settings_file=config_sparsity/DVS_base.yaml -e ",
    "python main.py -s sparsity/DVS/mink --settings_file=config_sparsity/DVS_mink.yaml -e ",
    "python main.py -s sparsity/DVS/mink_drop20 --settings_file=config_sparsity/DVS_mink_drop.yaml -e ",

    "python main.py -s sparsity/ASL/base --settings_file=config_sparsity/ASL_base.yaml -e ",
    "python main.py -s sparsity/ASL/mink --settings_file=config_sparsity/ASL_mink.yaml -e ",
    "python main.py -s sparsity/ASL/mink_drop20 --settings_file=config_sparsity/ASL_mink_drop.yaml -e ",

    "python main.py -s sparsity/NCal/base --settings_file=config_sparsity/NCal_base.yaml -e ",
    "python main.py -s sparsity/NCal/mink --settings_file=config_sparsity/NCal_mink.yaml -e ",
    "python main.py -s sparsity/NCal/mink_drop20 --settings_file=config_sparsity/NCal_mink_drop.yaml -e ",

    "python main.py -s sparsity/NMNIST/base --settings_file=config_sparsity/NMNIST_base.yaml -e ",
    "python main.py -s sparsity/NMNIST/mink --settings_file=config_sparsity/NMNIST_mink.yaml -e ",
    "python main.py -s sparsity/NMNIST/mink_drop20 --settings_file=config_sparsity/NMNIST_mink_drop.yaml -e ",

    "python main.py -s sparsity/IniRosh/base --settings_file=config_sparsity/IniRosh_base.yaml -e ",
    "python main.py -s sparsity/IniRosh/mink --settings_file=config_sparsity/IniRosh_mink.yaml -e ",
    "python main.py -s sparsity/IniRosh/mink_drop20 --settings_file=config_sparsity/IniRosh_mink_drop.yaml -e ",
    # "python main.py --settings_file=config_NAS/ASL/model_2929/3_ASL_settings_sgd_quant.yaml --dataset_percentage 0.1 --epoch 2 --fixBN_ratio 0.5 -s exp/simple_quant/ASL_2929",
    # "python main.py --settings_file=config_NAS/DVS/model_397/3_DVS_sgd_quant.yaml --dataset_percentage 0.1 --epoch 2 --fixBN_ratio 0.5 -s exp/simple_quant/NCal_397",

]

import os
for idx, cmd in enumerate(cmds):
    # cmd = cmd
    # cmd += " --json_file=model_json/DVS/model_{}.json".format(idx)
    print("Processing cmd {}".format(idx))
    os.system(cmd)