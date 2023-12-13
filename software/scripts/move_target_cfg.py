import os
import yaml

src_folder = "model_json/NCal"
cfg_yaml = "config/config_networkSearch/July23/NCal/1_NCAL_settings_sgd_baseline.yaml"

dataset = src_folder.split("/")[-1]
target_folder = "models/mobilenet_cfg_July23/" + dataset + "/"
os.makedirs(target_folder, exist_ok=True)
cfg_folder = "/".join(cfg_yaml.split("/")[:-1])
cmd_file = "cmd/{}_cmds.txt".format(dataset)
cmd_f = open(cmd_file, "w")

cfg_file = "model_json/{}_extract.csv".format(dataset)

with open(cfg_file, "r") as f:
    target_lines = f.readlines()[1:30]

target_files = [line.split(",")[0] for line in target_lines]


for file in target_files:
    with open(cfg_yaml, 'r') as f:
        lines = f.readlines()
    src_file = os.path.join(src_folder, file + ".json")
    target_file = os.path.join(target_folder, file + ".json")
    os.system("cp {} {}".format(src_file, target_file))
    yaml_target_file = os.path.join(cfg_folder, file + ".yaml")
    os.system("cp {} {}".format(cfg_yaml, yaml_target_file))
    # with open(cfg_yaml, 'r') as f:
    #     lines = f.readlines()
    lines[28] = lines[28].replace('model_type: ""', "model_type: {}".format(target_file))
    with open(yaml_target_file, 'w') as f:
        f.writelines(lines)
    # with open(cfg_yaml, 'r') as stream:
    #     cfg = yaml.load(stream, yaml.Loader)
    # cfg["model"]["model_type"] = target_file
    # yaml.safe_dump(cfg, open(os.path.join(cfg_folder, file + ".yaml"), "w"))
    cmd_f.write("python main.py --settings_file={} -s exp_NAS/{}_search/{}\n".
                format(yaml_target_file, src_folder.split("/")[-1], file))
    a = 1