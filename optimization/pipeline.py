
import os

HW_CONFIG_PATH = "/vol/datastore/EDSA/eventNetHWConfig"

model_path = "/vol/datastore/eye_tracking/eventModel"
config_path = "/vol/datastore/eye_tracking/eventHWConfig"
hw_path = "/vol/datastore/eye_tracking/eventNetHW"

hw = ["zcu102_80res", "zcu102_75res"] #, "zcu102_60res", "zcu102_80res", "zcu102_75res"
model = ["0324_cfg858_biasBit16"]

optim_cmd_sample = "python eventnet.py --model_path {} --hw_path {} --model_name {} --hw_name {} --results_path {}"
hw_prj_cmd_sample = "python gen_prj.py gen_full --cfg_name {} --cfg_path {} --tpl_dir template_e2e --dst_path {}"

optim_cmds, hw_prj_cmds = [], []

for h in hw:
    for m in model:
        optim_name = "{}-{}".format(m, h)
        optim_cmd = optim_cmd_sample.format(model_path, HW_CONFIG_PATH, m, h, config_path)
        hw_prj_cmd = hw_prj_cmd_sample.format(optim_name, config_path, hw_path)
        optim_cmds.append(optim_cmd)
        hw_prj_cmds.append(hw_prj_cmd)


for cmd in optim_cmds:
    print(cmd)

for cmd in hw_prj_cmds:
    print(cmd)

