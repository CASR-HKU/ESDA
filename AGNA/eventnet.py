import argparse
from copy import deepcopy
import json
import math
import os
import datetime
from formulation.eventnet_formulation import EventNetFormulation
from formulation.eventnet_formulation import (
    dsp_vfunc_dict,
    buf_vfunc_dict,
    buf_dscale_list,
    buf_dname_list,
)
from utils.my_logger import init_root_logger, get_logger
from utils.common import my_prod

legal = lambda name: name.replace(".", "_")
legal_low = lambda name: legal(name).lower()


def model_parser(model: dict) -> dict:
    """assign input_shape to each layer and verify the parameters"""
    logger = get_logger("")
    assert model["dataset"] in [
        "ASL",
        "DVS",
        "NCAL",
        "NMNIST",
        "Roshambo",
    ], f"Unknown dataset name: {model['dataset']}"
    parsed_model = deepcopy(model)
    prop_input_shape = parsed_model["input_shape"]
    for layer in parsed_model["layers"]:
        assert isinstance(layer["tensor_stride"], list)
        name = layer["name"]
        layer["name"] = legal_low(name)
        in_tensor_stride = layer["tensor_stride"][0]
        if layer["type"] == "conv":
            pass
        elif layer["type"] == "block":
            if layer["residual"] and layer["stride"] != 1:
                raise ValueError(f"{name}: residual block must have stride=1")
            if layer["residual"] and layer["channels"][0] != layer["channels"][2]:
                raise ValueError(f"{name}: residual block must have same channels")
        elif layer["type"] == "linear":
            in_tensor_stride = parsed_model["layers"][-2]["tensor_stride"][-1]
        else:
            raise ValueError("Unknown layer type")
        assert len(layer["kernel_sparsity"]) == 9
        # assign input_shape
        layer["input_shape"] = [
            math.ceil(parsed_model["input_shape"][0] / in_tensor_stride),
            math.ceil(parsed_model["input_shape"][1] / in_tensor_stride),
        ]
        logger.debug(f"{name} input_shape: {layer['input_shape']}")
        logger.debug(f"{name} input_shape(propagated): {prop_input_shape}")
        assert prop_input_shape == layer["input_shape"]
        prop_input_shape = [
            math.ceil(prop_input_shape[0] / layer["stride"]),
            math.ceil(prop_input_shape[1] / layer["stride"]),
        ]
    # check last layer
    last_layer = parsed_model["layers"][-1]
    nclass = last_layer["channels"][-1]
    if last_layer["type"] == "linear" and nclass in [25, 101]:
        last_layer["channels"][-1] = 2 ** math.ceil(math.log2(nclass))
        logger.warning(
            f"linear layer channels changed from {nclass} to {last_layer['channels'][-1]}"
        )
    assert last_layer["type"] == "linear"
    return parsed_model


def pre_solve_verify(model, hw):
    def verify_layer(name, ic, oc, ih, iw, ltype):
        logger = get_logger("")
        logger.debug(
            f"Verifying {name}: ic={ic}, oc={oc}, ih={ih}, iw={iw}, ltype={ltype}"
        )
        val_dict = {}
        dsp_cnt = 0
        bram_cnt = 0
        val_dict["pic"] = 2
        val_dict["poc"] = 2
        val_dict["tic"] = math.ceil(ic / val_dict["pic"])
        val_dict["toc"] = math.ceil(oc / val_dict["poc"])
        val_dict["iw"] = iw
        lsw = 8 + math.ceil(math.log2(ih * iw))
        logger.debug(f"{name} val_dict: {val_dict}")
        dsp_cnt = int(dsp_vfunc_dict[ltype](val_dict, "pic", "poc"))
        logger.debug(f"dsp_cnt: {dsp_cnt}")
        for bname in buf_vfunc_dict[ltype].keys():
            wdp = [
                math.ceil(
                    buf_vfunc_dict[ltype][bname][dname](
                        val_dict,
                        pic="pic",
                        poc="poc",
                        tic="tic",
                        toc="toc",
                        iw=iw,
                        linear_sum_w=lsw,
                    )
                    / buf_dscale_list[i]
                )
                for i, dname in enumerate(buf_dname_list)
            ]
            bram_cnt += int(my_prod(wdp))
            logger.debug(f"bram_cnt: {bram_cnt} ({bname} wdp: {wdp})")
        return dsp_cnt, bram_cnt

    total_dsp_cnt = 0
    total_bram_cnt = 0
    for layer in model["layers"]:
        if layer["type"] == "conv" or layer["type"] == "linear":
            ltype = (
                "conv_1x1"
                if layer["name"] == "conv8"
                else "conv_3x3"
                if layer["name"] == "conv1"
                else "linear"
                if layer["type"] == "linear"
                else None
            )
            dsp_cnt, bram_cnt = verify_layer(
                layer["name"], *layer["channels"], *layer["input_shape"], ltype
            )
            total_dsp_cnt += dsp_cnt
            total_bram_cnt += bram_cnt
        elif layer["type"] == "block":
            chs = layer["channels"][:2]
            ips = layer["input_shape"]
            dsp_cnt, bram_cnt = verify_layer(
                layer["name"] + "_c0", *chs, *ips, "conv_1x1"
            )
            total_dsp_cnt += dsp_cnt
            total_bram_cnt += bram_cnt
            chs = [layer["channels"][1]] * 2
            dsp_cnt, bram_cnt = verify_layer(
                layer["name"] + "_c1", *chs, *ips, "conv_3x3_dw"
            )
            total_dsp_cnt += dsp_cnt
            total_bram_cnt += bram_cnt
            chs = layer["channels"][1:]
            ips = [x // layer["stride"] for x in ips]
            dsp_cnt, bram_cnt = verify_layer(
                layer["name"] + "_c2", *chs, *ips, "conv_1x1"
            )
            total_dsp_cnt += dsp_cnt
            total_bram_cnt += bram_cnt
        else:
            raise ValueError("Unknown layer type")
    logger = get_logger("")
    logger_f = logger.warning if total_dsp_cnt > hw["dsp"] else logger.info
    logger_f(f"total_dsp: {total_dsp_cnt}/{hw['dsp']}")
    logger_f = logger.warning if total_bram_cnt > hw["bram36"] * 2 else logger.info
    logger_f(f"total_bram: {total_bram_cnt}/{hw['bram36']*2}")


config_list = [
    {
        "form-obj": "lat_max+uti",
        "form-fix_top_pc": True,
        "scip-use_cmd": True,
        "rnr-range": 1.5,
        "scip-timelimits": 900,
    },
    {
        "form-obj": "lat_max+uti",
        "form-fix_top_pc": True,
        "scip-use_cmd": True,
        "rnr-range": 2,
        "scip-timelimits": 900,
    },
    {
        "form-obj": "lat_max",
        "scip-use_cmd": True,
        "rnr-range": 1.5,
        "scip-timelimits": 900,
    },
    {
        # "solver-flow": "scip",
        "form-obj": "lat_max",
        "form-fix_top_pc": True,
        "form-fix_layer_pc": 2,
        "scip-use_cmd": True,
        "rnr-range": 1.5,
        "scip-timelimits": 900,
    },
]


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/vol/datastore/eventNetModel/0727_NASModel"
    )
    parser.add_argument("--model_name", type=str, default="DVS_1890")
    parser.add_argument(
        "--hw_path", type=str, default="/vol/datastore/eventNetHWConfig"
    )
    parser.add_argument("--hw_name", type=str, default="zcu102_80res")
    parser.add_argument(
        "--results_path", type=str, default="/vol/datastore/eventNetConfig/default_runs"
    )
    parser.add_argument("--config", type=int, default=0)
    parser.add_argument("--debug", type=str, help="debug message")
    parser.add_argument("--nas", action="store_true", default=False)
    args = parser.parse_args()

    model_name = args.model_name
    hw_name = args.hw_name
    model_hw_name = f"{model_name}-{hw_name}"
    config_idx = args.config
    if args.debug:
        model_hw_name += f"-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        model_hw_name += f"-config{config_idx}"
    work_path = os.path.join(args.results_path, model_hw_name)
    if not os.path.isdir(work_path):
        os.makedirs(work_path, exist_ok=True)

    # get logger
    logger = init_root_logger(os.path.join(work_path, f"main.log"))
    # load model and hw
    if args.nas:
        model_cfg_path = os.path.join(args.model_path, f"{model_name}.json")
    else:
        model_cfg_path = os.path.join(args.model_path, model_name, "model.json")
    model_cfg = json.load(open(model_cfg_path))
    logger.info(f"Model config loaded from {model_cfg_path}")
    hw_cfg_path = os.path.join(args.hw_path, f"{hw_name}.json")
    hw_cfg = json.load(open(hw_cfg_path))
    logger.info(f"HW config loaded from {hw_cfg_path}")

    # model preprocessing
    model_cfg = model_parser(model_cfg)
    pre_solve_verify(model_cfg, hw_cfg)
    config = config_list[config_idx]
    if args.debug:
        config["message"] = args.debug
    # formulation and solve
    try:
        en = EventNetFormulation("en", work_path, model_cfg, hw_cfg, config)
        en.solve()
    except Exception as e:
        logger.exception(e)
    finally:
        json.dump(
            en.config, open(os.path.join(work_path, "en-config.json"), "w"), indent=4
        )
    if en.result is not None:
        result_path = os.path.join(work_path, "en-result.json")
        en.result["model_path"] = f"{args.model_path}/{model_name}"  # for loading .npy
        # save_result(en.result, result_path)
        json.dump(en.result, open(result_path, "w"), indent=4)
        logger.info(f"overall obj: {en.result['obj']}")
        logger.info(f"overall dsp: {en.result['total_dsp']}")
        logger.info(f"overall dsp_uti: {en.result['total_dsp']/hw_cfg['dsp']:.2f}")
        logger.info(f"overall bram: {en.result['total_bram']}")
        logger.info(
            f"overall bram_uti: {en.result['total_bram']/(hw_cfg['bram36']*2):.2f}"
        )
        logger.info(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
