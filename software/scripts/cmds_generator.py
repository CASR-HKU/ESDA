import os


data_path = "Documents/event_dataset_preprocessed/dvs_gesture_clip"
file_name = "DVS_cmd.txt"


def get_ratio(cfg):
    if "0p1" in cfg:
        return "0p1"
    elif "0p2" in cfg:
        return "0p2"
    elif "0p3" in cfg:
        return "0p3"
    else:
        raise ValueError("No ratio found in {}".format(cfg))


def get_operation(cfg):
    if "absSum" in cfg:
        return "absSum"
    elif "random" in cfg:
        return "random"
    else:
        raise ValueError("No operation found in {}".format(cfg))


def generate_baseline(cfg_folder, exp_folder, epochs=180, file="cmd.txt"):
    sample_size = [5000, 10000]
    baselines_cfg = [file for file in os.listdir(cfg_folder) if "1_" in file]
    with open(file, "a+") as f:
        for size in sample_size:
            for baseline_cfg in baselines_cfg:
                random = "" if "random" not in baseline_cfg else "_random"
                output_folder = os.path.join(exp_folder, "sample{}".format(size) + random)
                cmd = "python3 main.py --settings_file={} --s {} --epochs {}".\
                    format(os.path.join(cfg_folder, baseline_cfg), output_folder, epochs)
                if data_path:
                    cmd += " --data_path {}".format(data_path)
                f.write(cmd + "\n")
        f.write("\n")


def generate_drop_or_quant(cfg_folder, exp_folder, epochs=180, baseline_file="", file="cmd.txt", generally=False):

    single_cfg = [file for file in os.listdir(cfg_folder) if "2_" in file]
    with open(file, "a+") as f:
        for cfg in single_cfg:
            if "quant" in cfg:
                output_folder = os.path.join(exp_folder, "quant")
                quant_epoch = int(epochs * 2/3)
                for shift in [31, 15, 7]:
                    output_folder += str(shift)
                    cmd = "python3 main.py --settings_file={} --s {} --epochs {} --load_file {} --fixBN_ratio 0.3". \
                        format(os.path.join(cfg_folder, cfg), output_folder, quant_epoch, baseline_file)
                    cmd += " --shift_bit {}".format(shift)
                    if data_path:
                        cmd += " --data_path {}".format(data_path)
                # for shift in [31, 15, 7]:
                #     cmd += " --shift_bit {}".format(shift)
                        f.write(cmd + "\n")
            else:
                drop_epoch = int(epochs * 5/6)
                folder_name = get_operation(cfg) + "_" + get_ratio(cfg)
                output_folder = os.path.join(exp_folder, folder_name)
                if generally:
                    output_folder += "_generally --generally "
                cmd = "python3 main.py --settings_file={} -s {} --epochs {} --load {}".\
                    format(os.path.join(cfg_folder, cfg), output_folder, drop_epoch, baseline_file)
                if data_path:
                    cmd += " --data_path {}".format(data_path)
                f.write(cmd + "\n")
        f.write("\n")


def generate_drop_and_quant(cfg_folder, exp_folder,  epochs=180, shift=15, file="cmd.txt", generally=False):
    drop_quant_cfgs = [file for file in os.listdir(cfg_folder) if "3_" in file]
    with open(file, "a+") as f:
        for cfg in drop_quant_cfgs:
            cfg_path = os.path.join(cfg_folder, cfg)
            target_epoch = int(epochs * 2/3)
            exp_folder_name = get_operation(cfg) + "_" + get_ratio(cfg)
            ratio, operation = get_ratio(cfg), get_operation(cfg)
            if generally:
                load_file = exp_folder + "/{}_{}_generally".format(operation, ratio) + "/ckpt.best.pth.tar"
                exp_folder_name += "_generally"
            else:
                load_file = exp_folder + "/{}_{}".format(operation, ratio) + "/ckpt.best.pth.tar"
            exp_folder_name += "_quant_shift{}".format(shift)
            output_folder = os.path.join(exp_folder, exp_folder_name)
            cmd = "python3 main.py --settings_file={} -s {} --epochs {} --load {} --shift_bit {} " \
                  "--fixBN_ratio 0.3".format(cfg_path, output_folder, target_epoch, load_file, shift)
            if data_path:
                cmd += " --data_path {}".format(data_path)
            f.write(cmd + "\n")
        f.write("\n")


if __name__ == '__main__':
    generally = False
    cfg_folder = "config/config_with0p5/0710_DVS"
    exp_folder = "exp_NAS/DVS_width0p5"
    epochs = 180

    # generate_baseline(cfg_folder, exp_folder, epochs, file_name)

    load = "~/Downloads/DVS_searched/baseline_0p5/ckpt.best.pth.tar"
    generate_drop_or_quant(cfg_folder, exp_folder, epochs, load, file_name, generally)

    shift = 15
    generate_drop_and_quant(cfg_folder, exp_folder, epochs, shift, file_name, generally)

