import os
import shutil


# group_name = "ASL_0p5_shift16"
src_folder = "/vol/datastore/baoheng/eventModel/EDSA_correctShift"
# group_names = ["NMNIST_shift16"]
group_names = os.listdir(src_folder)
gen_folder = src_folder + "_modified"
origin_final_folder = "/vol/datastore/baoheng/eventModel/EDSA_correctShift_raw"

for group_name in group_names:
    shift_bit = int(group_name[-2:])
    src_path_npy = "/vol/datastore/baoheng/eventModel/EDSA_correctShift/{}".format(group_name)
    src_path_hw = "/vol/datastore/baoheng/eventNetHW/EDSA_correctShift/{}-zcu102_80res/full".format(group_name)

    gen_path_npy = "tmp"
    final_npy_path = src_path_npy.replace("EDSA_correctShift", "EDSA_correctShift_modified")
    os.makedirs(gen_path_npy, exist_ok=True)
    os.makedirs(final_npy_path, exist_ok=True)

    cfg_path = os.path.join(src_path_hw, 'cfg.json')
    cfg_dest_path = os.path.join(src_path_npy, 'cfg.json')
    cfg_tmp_path = os.path.join(gen_path_npy, 'cfg.json')
    shutil.copyfile(cfg_path, cfg_tmp_path)
    shutil.copyfile(cfg_path, cfg_dest_path)

    cmd = "python sw_e2e.py -d {} -s {} --shift_bit {}".format(src_path_npy, gen_path_npy, shift_bit)
    print(cmd)
    os.system(cmd)
    cmd = "python gen_data.py -d {} --shift_bit {} --same_folder".format(gen_path_npy, shift_bit)
    print(cmd)
    os.system(cmd)
    move_cmd = "mv {}/* {}".format(gen_path_npy, final_npy_path)
    print(move_cmd)
    os.system(move_cmd)
    # shutil.move(gen_path_npy, final_npy_path)

# shutil.move(src_folder, origin_final_folder)
# shutil.move(gen_folder, src_folder)
