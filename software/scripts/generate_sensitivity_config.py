import os
import subprocess


file = "txt/randomDropWith.txt"
folder = "config/config_random_sensitivity/with"
load = "~/Downloads/DVS_searched/baseline_0p5/ckpt.best.pth.tar"

f = open(file, "w")
f.write("Group\nAcc\n")

configs = os.listdir(folder)

# for cmd in cmds:
for config in configs:
    cfg = '_'.join(config.split(".")[0].split("_")[-2:])
    cmd = "python main.py -e --settings_file={} --load {}".format(os.path.join(folder, config), load)
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
    acc = output.split("\n")[-2].split(" ")[0]
    f.write("{}\n{}\n".format(cfg, acc))
    print(output)


