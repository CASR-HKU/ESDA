import json
import os
import csv
import numpy as np
import xml.etree.ElementTree as ET

benchmark_path = "benchmark_results"
cosim_sparse_nzr_list = ["0.1", "0.2", "0.4", "0.8", "1.0"]
# cosim_sparse_nzr_list = ["0.1", "0.3", "0.5", "0.7", "0.9"]
cosim_dense_nzr_list = ["1"]
evaluate_sparse_nzr_list = ["0.1", "0.2", "0.4", "0.8"]
evaluate_dense_nzr_list = ["1"]
key_list = (
    [f"cosim_sparse_{r}" for r in cosim_sparse_nzr_list]
    + [f"cosim_dense_{r}" for r in cosim_dense_nzr_list]
    + [f"evaluate_sparse_{r}" for r in evaluate_sparse_nzr_list]
    + [f"evaluate_dense_{r}" for r in evaluate_dense_nzr_list]
)


def load_cosim_log(dir, nzr_list):
    results = [None] * len(nzr_list)
    log_file = os.path.join(
        dir, "prj/sp_hls_new_proj/solution1/sim/report/verilog/result.transaction.rpt"
    )
    if not os.path.exists(log_file):
        return results
    with open(log_file, "r") as f:
        lines = [line for line in f.readlines() if line.startswith("transaction")]
        for i, line in zip(range(len(nzr_list)), lines):
            results[i] = int(line[20:-12].strip())
    return results


def load_evaluate_log(dir, nzr_list):
    results = [None] * len(nzr_list)
    log_file = os.path.join(dir, "hw", "evaluate.log")
    if not os.path.exists(log_file):
        return results
    with open(log_file, "r") as f:
        lines = f.readlines()
        for i in range(len(nzr_list)):
            beg_line = lines.index(f"====BEGIN with {nzr_list[i]}\n")
            if lines[beg_line + 1].startswith("nz_ratio"):
                num_run = int(lines[beg_line + 1].split("=")[-1])
            else:
                num_run = 0
            if lines[beg_line + 2].startswith("average"):
                runtime = float(lines[beg_line + 2].split(":")[-1])
            else:
                runtime = 0
            if lines[beg_line + 3].startswith("====RETURN"):
                retcode = int(lines[beg_line + 3].split(" ")[-1])
            else:
                retcode = None
            results[i] = runtime
    return results


def loop_sub_dirs(path, cosim_keys, evaluate_keys):
    results = {}
    # loop all subdirs in path
    for blk in os.listdir(path):
        dir = os.path.join(path, blk)
        if os.path.isdir(dir) and blk.startswith("single"):
            results[blk] = {}
            results[blk]["cosim"] = load_cosim_log(dir, cosim_keys)
            results[blk]["evaluate"] = load_evaluate_log(dir, evaluate_keys)
    return results


def extract_csv_rows(cfg_name):
    sparse_path = f"{benchmark_path}/sparse/{cfg_name}"
    sparse_results = loop_sub_dirs(
        sparse_path, cosim_sparse_nzr_list, evaluate_sparse_nzr_list
    )
    dense_path = f"{benchmark_path}/dense/{cfg_name}"
    dense_results = loop_sub_dirs(
        dense_path, cosim_dense_nzr_list, evaluate_dense_nzr_list
    )
    # build csv rows
    csv_rows = []
    for blk in sorted(sparse_results.keys(), key=lambda s: int(s.split("_")[-1])):
        csv_row = {"name": blk}
        # print(sparse_results[blk] + dense_results[blk])
        csv_row.update(
            {
                k: v
                for k, v in zip(
                    key_list,
                    sparse_results[blk]["cosim"]
                    + dense_results[blk]["cosim"]
                    + sparse_results[blk]["evaluate"]
                    + dense_results[blk]["evaluate"],
                )
            }
        )
        cfg_json = json.load(open(f"{dense_path}/{blk}/cfg.json", "r"))
        csv_row["parallelism"] = cfg_json["layers"][0]["parallelism"]
        csv_row["stride"] = cfg_json["layers"][0]["stride"]
        sparse_csynth_xml = (
            f"{dense_path}/{blk}/prj/sp_hls_new_proj/solution1/syn/report/csynth.xml"
        )
        csv_row["sparse_bram"] = (
            ET.parse(sparse_csynth_xml).find("AreaEstimates/Resources/BRAM_18K").text
        )
        dense_csynth_xml = (
            f"{dense_path}/{blk}/prj/sp_hls_new_proj/solution1/syn/report/csynth.xml"
        )
        csv_row["dense_bram"] = (
            ET.parse(dense_csynth_xml).find("AreaEstimates/Resources/BRAM_18K").text
        )
        csv_rows.append(csv_row)
    return csv_rows


def save_csv_rows(csv_rows, cfg_name):
    # write csv
    csv_file = os.path.join(f"{benchmark_path}/{cfg_name}.csv")
    with open(csv_file, "w") as f:
        fields = (
            ["name"]
            + key_list
            + [
                "parallelism",
                "stride",
                "sparse_bram",
                "dense_bram",
            ]
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()}
            for row in csv_rows
        )
    # save as npy
    npy_file = os.path.join(f"{benchmark_path}/{cfg_name}.npy")
    np_arr = np.array([[r[k] for k in key_list] for r in csv_rows])
    np.save(npy_file, np_arr)
    print(f"csv saved to {csv_file}")
    print(f"npy saved to {npy_file}")


def main():
    csv_rows0 = extract_csv_rows("DVS_mobilenet_0707_0p5-zcu102_50res_new")
    save_csv_rows(csv_rows0, "DVS_mobilenet_0707_0p5-zcu102_50res_new")
    # csv_rows1 = extract_csv_rows("new_manual_config")
    # save_csv_rows(csv_rows1, "new_manual_config")
    # for row1 in csv_rows1:
    #     for row0 in csv_rows0:
    #         if row0["name"] == row1["name"]:
    #             row0.update(row1)
    #             break
    # save_csv_rows(csv_rows0, "extract")


if __name__ == "__main__":
    main()
