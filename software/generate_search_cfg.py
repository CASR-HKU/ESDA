import json
import os

json_folder = "configs/search_fixRNN128_2080"
output_folder = "configs/0322_submanifold_search_fixedRNN128_2080"
os.makedirs(output_folder, exist_ok=True)
cmd_generate_folder = output_folder.split("/")[1]
cfg_template = "configs/0227_base_submanifold/mobilenet_submanifold_length30.json"
cmds_path = os.path.join(output_folder, "cmds.txt")
cmds_file = open(cmds_path, "w")

for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        output_json = os.path.join(output_folder, filename)
        with open(cfg_template, 'r') as ft:
            cfg = json.load(ft)
            cfg["model_cfg"] = os.path.join(json_folder, filename)
            cfg["mlflow_path"] = f"exp/{filename.split('.')[0]}"
            with open(f"{output_folder}/{filename}", 'w') as fw:
                json.dump(cfg, fw, indent=4)
                # print(f"configs/0214_mobilenet/{filename} generated.")
        cmd = ("CUDA_VISIBLE_DEVICES=0 python train.py --config_file={} --mlflow_path=exp/{}/{} --batch_size 8".
               format(output_json, cmd_generate_folder, filename.split('.')[0]))
        cmds_file.write("'" + cmd + "',\n")
