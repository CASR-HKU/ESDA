import os

src_path = '/vol/datastore/baoheng/eventNetHW/EDSA_correctShift'
dest_path = '/vol/datastore/baoheng/eventModel/EDSA_correctShift_modified'

folders = os.listdir(src_path)
for folder in folders:
    if "Makefile" in folder:
        continue
    t_folder_name = folder.split("-")[0]
    os.makedirs(os.path.join(dest_path, t_folder_name), exist_ok=True)
    cmd = "python sw_e2e.py -d {} -s {}".format(os.path.join(src_path, folder, "full"),
                                                os.path.join(dest_path, t_folder_name))
    # print(cmd)
    os.system(cmd)
