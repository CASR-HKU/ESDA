
import os
import json
import numpy as np


layer_sparsity = []
kernel_sparsity = []
input_shape = [180, 240]

src_json_folder = "model_NAS/model_json_1w"
json_files = os.listdir(src_json_folder)
target_folder = "model_NAS/model_json_1w_NCal"

for json_file in json_files:
    with open(os.path.join(src_json_folder, json_file), "r") as f:
        model = json.loads(f.read())
        model["input_shape"] = input_shape
        model["input_sparsity"] = layer_sparsity[0]
        model["layers"][0]["sparsity"] = layer_sparsity[1]
        model["layers"][0]["kernel_sparsity"] = kernel_sparsity[1]
        model["layers"][-1]["sparsity"] = layer_sparsity[-1]
        model["layers"][-1]["kernel_sparsity"] = kernel_sparsity[-1]
        res = 1
        for layers in model["layers"][1:-1]:
            layers["kernel_sparsity"] = [kernel_sparsity[res]]
            if layers["stride"] == 2:
                layers["sparsity"] = [layer_sparsity[res], layer_sparsity[res+1], layer_sparsity[res+1]]
                res += 1
            else:
                layers["sparsity"] = [layer_sparsity[res] for _ in range(3)]

        kernel_sparsity.append(model["kernel_sparsity"])
        json.dump(model, open(os.path.join(target_folder, json_file), "w"))
        # layer_sparsity.append(model["layer_sparsity"])
        # kernel_sparsity.append(model["kernel_sparsity"])

