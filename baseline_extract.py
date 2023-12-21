import argparse
import datetime
import numpy as np
import csv
import os
import json
import xml.etree.ElementTree as ET

pmbus_unit = {
    # PS 10 sensors
    "VCCPSINTFP": "W",
    "VCCPSINTLP": "mW",
    "VCCPSAUX": "mW",
    "VCCPSPLL": "mW",
    "MGTRAVCC": "mW",
    "MGTRAVTT": "mW",
    "VCCO_PSDDR_504": "mW",
    "VCCOPS": "W",
    "VCCOPS3": "W",
    "VCCPSDDRPLL": "mW",
    # PL 8 sensors
    "VCCINT": "W",
    "VCCBRAM": "mW",
    "VCCAUX": "mW",
    "VCC1V2": "mW",
    "VCC3V3": "mW",
    "VADJ_FMC": "W",
    "MGTAVCC": "mW",
    "MGTAVTT": "mW",
}


def extract_full(path, result: dict):
    # load cfg
    cfg_json = os.path.join(path, "cfg.json")
    result.update({"cfg_dsp": None, "cfg_bram": None})
    if os.path.exists(cfg_json):
        cfg = json.load(open(cfg_json, "r"))
        result["cfg_obj"] = cfg["obj"]
        result["cfg_dsp"] = cfg["total_dsp"]
        result["cfg_bram"] = cfg["total_bram"]
    # load csynth rpt
    csynth_xml = os.path.join(
        path, "prj/sp_hls_new_proj/solution1/syn/report/csynth.xml"
    )
    result.update({"csynth_dsp": None, "csynth_bram": None})
    if os.path.exists(csynth_xml):
        tree = ET.parse(csynth_xml)
        root = tree.getroot()
        result["csynth_dsp"] = int(root.find("AreaEstimates/Resources/DSP").text)
        result["csynth_bram"] = int(root.find("AreaEstimates/Resources/BRAM_18K").text)
    # load cosim rpt
    cosim_rpt = os.path.join(
        path, "prj/sp_hls_new_proj/solution1/sim/report/verilog/result.transaction.rpt"
    )
    result.update({"cosim_lat": None})
    if os.path.exists(cosim_rpt):
        line = open(cosim_rpt, "r").readlines()[1]
        result["cosim_lat"] = int(line[20:37].strip())
    # load vivado power rpt
    vpower_rpt = os.path.join(
        path,
        "prj/sp_new_vivado_proj/sp_new_vivado_proj.runs/impl_1/design_1_wrapper_power_routed.rpt",
    )
    result.update({"vpower_static": None, "vpower_dynamic": None})
    if os.path.exists(vpower_rpt):
        lines = open(vpower_rpt, "r").readlines()
        static_line = lines[36]
        dynamic_line = lines[35]
        result["vpower_static"] = float(static_line.split("|")[2].strip())
        result["vpower_dynamic"] = float(dynamic_line.split("|")[2].strip())
    # load vivado timing rpt
    vtiming_rpt = os.path.join(
        path,
        "prj/sp_new_vivado_proj/sp_new_vivado_proj.runs/impl_1/design_1_wrapper_timing_summary_routed.rpt",
    )
    result.update({"vtiming_wns": None})
    if os.path.exists(vtiming_rpt):
        lines = open(vtiming_rpt, "r").readlines()
        wns_line = lines[131]
        result["vtiming_wns"] = float(wns_line[:12].strip())
    # load vivado impl rpt
    vimpl_rpt = os.path.join(
        path,
        "prj/sp_new_vivado_proj/sp_new_vivado_proj.runs/impl_1/design_1_wrapper_utilization_placed.rpt",
    )
    result.update({"vimpl_dsp": None, "vimpl_bram": None})
    if os.path.exists(vimpl_rpt):
        rpt_lines = open(vimpl_rpt, "r").readlines()
        dsp_line = rpt_lines[121]  # DSPs
        bram_line = rpt_lines[106]  # Block RAM Tile
        ff_line = rpt_lines[34]  # CLB LUTs
        lut_line = rpt_lines[39]  # CLB Registers
        result["vimpl_dsp"] = int(dsp_line.split("|")[2].strip())
        result["vimpl_bram"] = int(float(bram_line.split("|")[2].strip()) * 2)
        result["vimpl_ff"] = f"{round(int(ff_line.split('|')[2].strip()) / 1000)}K"
        result["vimpl_lut"] = f"{round(int(lut_line.split('|')[2].strip()) / 1000)}K"
    # else:
    #     print(f"vimpl_rpt not found: {vimpl_rpt}")
    # load hw/evaluate.log
    hw_eval_log = os.path.join(path, "hw/evaluate.log")
    result.update(
        {
            "hw_eval_log_mtime": None,
            "hw_dataset": None,
            "hw_num_run": None,
            "hw_start": None,
            "hw_end": None,
            "hw_duration": None,
            "hw_runtime": None,
            "pm_start": None,
            "pm_end": None,
            "pm_duration": None,
            "pm_avgtime": None,
            "hw_retcode": None,
        }
    )
    if os.path.exists(hw_eval_log):
        lines = open(hw_eval_log, "r").readlines()
        mtime_stamp = os.path.getmtime(hw_eval_log)
        mtime_str = datetime.datetime.fromtimestamp(mtime_stamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        result["hw_eval_log_mtime"] = mtime_str
        begin_line = next(
            filter(lambda line: line.startswith("====BEGIN"), lines), ""
        ).removesuffix("\n")
        dtst_line = next(
            filter(lambda line: line.startswith("dataset="), lines), ""
        ).removesuffix("\n")
        hwlp_line = next(
            filter(lambda line: line.startswith("hw loop"), lines), ""
        ).removesuffix("\n")
        hwrt_line = next(
            filter(lambda line: line.startswith("hw average runtime"), lines), ""
        ).removesuffix("\n")
        pmlp_line = next(
            filter(lambda line: line.startswith("pm time"), lines), ""
        ).removesuffix("\n")
        end_line = next(
            filter(lambda line: line.startswith("====RETURN"), lines), ""
        ).removesuffix("\n")
        if dtst_line != "":
            result["hw_dataset"] = dtst_line.split(",")[0].split("=")[-1]
            result["hw_num_run"] = dtst_line.split(",")[1].split("=")[-1]
        if hwlp_line != "":
            result["hw_start"] = hwlp_line.split(": ")[-1].split(" ")[2]
            result["hw_end"] = hwlp_line.split(": ")[-1].split(" ")[0]
            result["hw_duration"] = hwlp_line.split(": ")[-1].split(" ")[4]
        if hwrt_line != "":
            hwrt = float(hwrt_line.split(": ")[-1])
            result["hw_runtime"] = f"{hwrt:.2f}"
            result["hw_throughput"] = f"{int(1000 / hwrt)}"
        if pmlp_line != "":
            result["pm_start"] = pmlp_line.split(": ")[-1].split(" ")[2]
            result["pm_end"] = pmlp_line.split(": ")[-1].split(" ")[0]
            result["pm_duration"] = pmlp_line.split(": ")[-1].split(" ")[4]
            result["pm_avgtime"] = pmlp_line.split(": ")[-1].split(" ")[8]
        if end_line != "":
            result["hw_retcode"] = end_line.split(" ")[-1]
    # load hw/power_record.npy
    hw_power_record = os.path.join(path, "hw/power_record.npy")
    result.update({"hw_power_range": None})
    result.update({"hw_power_statistic": None})
    if os.path.exists(hw_power_record):
        hw_pr = np.load(hw_power_record)
        # find index of first hw_pr[?, 0] >= result["hw_start"]
        hw_start_idx = np.argmax(hw_pr[:, 0] >= float(result["hw_start"]))
        # find index of first hw_pr[?, 0] >= result["hw_end"]
        hw_end_idx = (
            hw_pr.shape[0] - 1 - np.argmax(hw_pr[::-1, 0] >= float(result["hw_end"]))
        )
        hw_average_pr = np.average(hw_pr[hw_start_idx:hw_end_idx, 1:], axis=0)
        power_table = hw_average_pr.reshape((18, 4))[:, 2]
        ps_power = np.sum(power_table[:10])
        pl_power = np.sum(power_table[10:])
        result["hw_power_range"] = [hw_start_idx, hw_end_idx]
        result["hw_power_statistic"] = [f"{v:.04f}" for v in power_table]
        result["hw_power_ps"] = ps_power
        result["hw_power_pl"] = f"{pl_power:.2f}"
        result["hw_power_pl_eff"] = f"{pl_power*hwrt:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_path", type=str)
    parser.add_argument("--extract_large", action="store_true")
    args = parser.parse_args()
    runs_path = args.runs_path

    runs_result = []
    # dir_list = sorted(os.listdir(runs_path))
    if args.extract_large:
       dir_list = [
           "NCal_2751_shift32-zcu102_80res",
           "NCal_w0p5_shift32-zcu102_50res",
           "DVS_1890_shift16-zcu102_80res",
           "DVS_0p5_shift16-zcu102_60res",
           "ASL_2929_shift16-zcu102_80res",
           "ASL_0p5_shift16-zcu102_80res",
           "NMNIST_shift16-zcu102_60res",
           "Roshambo_shift16-zcu102_80res"
       ]
    for subdir in dir_list:
        run_path = os.path.join(runs_path, subdir, "full")
        run_result = {"name": subdir}
        if not os.path.exists(f"{run_path}/prj/ip_repo"):
            run_result["status"] = "hls failed"
        elif not os.path.exists(f"{run_path}/hw"):
            run_result["status"] = "vivado failed"
        else:
            run_result["status"] = ""
        extract_full(run_path, run_result)
        runs_result.append(run_result)
    write_k_list = [
        "name",
        "vpower_static",
        "vpower_dynamic",
        "status",
        "hw_num_run",
        "hw_dataset",
        "hw_runtime",
        "cosim_lat",
        "hw_duration",
        "pm_duration",
        "pm_avgtime",
        "hw_power_range",
        "hw_power_ps",
        "hw_power_pl",
    ]
    # write_k_list = runs_result[0].keys()
    if args.extract_large:
        write_k_list = [
            "name",
            "hw_num_run",
            "hw_eval_log_mtime",
            "cfg_obj",
            "cosim_lat",
            "vtiming_wns",
            "vimpl_dsp",
            "vimpl_bram",
            "vimpl_ff",
            "vimpl_lut",
            "hw_runtime",
            "hw_throughput",
            "hw_power_pl",
            "hw_power_pl_eff",
        ]
        csv_file = os.path.join(runs_path, "extract_large.csv")
    else:
        csv_file = os.path.join(runs_path, "extract.csv")
    with open(csv_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=write_k_list, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs_result)
    print(f"write to {csv_file}")


if __name__ == "__main__":
    main()
