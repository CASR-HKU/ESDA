import os
import json

src_csv = "model_json/DVS_extract.csv"
src_folder = "model_json/DVS_new"
dest_csv = src_csv.replace(".csv", "_addParam.csv")


with open(src_csv, "r") as f:
    lines = f.readlines()

target_line = [lines[0][:-1] + ",param\n"]
for line in lines[1:]:
    src_file = line.split(",")[0]
    json_file = os.path.join(src_folder, src_file) + ".json"
    with open(json_file, "r") as f:
        model = json.loads(f.read())
        target_line.append(line[:-1] + "," + str(model["param"]) + "\n")
        a = 1

with open(dest_csv, "w") as f:
    f.writelines(target_line)
    a = 1
