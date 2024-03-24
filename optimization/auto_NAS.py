import os

src_folder = "/vol/datastore/eye_tracking/NAS/0323_src"
optimed_folder = "/vol/datastore/eye_tracking/NAS/0323_optim"

cmd_base = ("python eventnet.py --nas --model_path {} --hw_path /vol/datastore/EDSA/eventNetHWConfig "
            "--model_name {} --hw_name zcu102_80res --results_path {}")

for file in os.listdir(src_folder):
    model_name = file.split(".")[0]
    cmd = cmd_base.format(src_folder, model_name, optimed_folder)
    os.system(cmd)
    print(cmd)

