from copy import deepcopy
import math
import os
import json
import argparse
import shutil
import warnings

from common import *

dflt_top_name = "top"
valid_dataset_list = ["EyeTracking", "ASL", "DVS", "NCAL", "NMNIST", "Roshambo"]


def single_fifo_code_block(fifo_name, dtype, depth):
    return [
        f"    hls::stream<{dtype}> {fifo_name};\n",
        f"#pragma HLS STREAM variable={fifo_name} depth={depth}\n",
    ]


def func_code_block(func_name, tpl_list, arg_list):
    tpl_str = ", ".join(tpl_list)
    arg_str = ", ".join(arg_list)
    return [
        f"    {func_name}<{tpl_str}>({arg_str});\n",
    ]


def para_code_block(def_dict):
    return [f"#define {k} {v}\n" for k, v in def_dict.items()]


def variable_code_block(name, t, width, depth):
    vname = globals()[f"{t}buf_of"](name)
    fname = globals()[f"{t}file_of"](name)
    if depth is None:
        declare_str = f"const ap_int<{width}> {vname}"
    else:
        declare_str = f"const ap_int<{width}> {vname}[{depth}]"
    return [
        f"{declare_str} = {{\n",
        f'    #include "data/{fname}.txt"\n',
        "};\n",
    ]


def fifo_code_block(name, cfg="POC"):
    fifo_list = []
    fifo_list += single_fifo_code_block(
        afifo_of(name), f"BundleT<{cfg_of(name, cfg)}, ap_int<CFG_AW>>", 2
    )
    fifo_list += single_fifo_code_block(tfifo_of(name), "T_K", 2)
    return fifo_list


def append_prolouge_code(codes, common_para):
    """append prolouge code to codes, including init_fifo, load, para.h, weight.h"""
    # define init fifo
    codes["fifo"] += fifo_code_block(dflt_top_name, "PIC")
    # fifo for mask
    codes["fifo"] += single_fifo_code_block(
        mfifo_of(dflt_top_name), "ap_int<CFG_MW>", 20
    )
    codes["fifo"] += single_fifo_code_block(ts2fifo_of(dflt_top_name), "T_K", 128)
    # load to act_fifo
    codes["load"] += func_code_block(
        "read_sparse_input",
        [cfg_of(dflt_top_name, "PIC"), "CFG_AW", cfg_of(dflt_top_name, "IC")],
        ["act_in", afifo_of(dflt_top_name), "num_nz"],
    )
    # load to token_fifo
    codes["load"] += func_code_block(
        "M2S_mask",
        ["CFG_MW", cfg_of(dflt_top_name, "IH"), cfg_of(dflt_top_name, "IW")],
        ["mask", tfifo_of(dflt_top_name), mfifo_of(dflt_top_name)],
    )
    codes["load"] += func_code_block(
        "mask_stride2",
        ["CFG_MW", cfg_of(dflt_top_name, "IH"), cfg_of(dflt_top_name, "IW")],
        [mfifo_of(dflt_top_name), ts2fifo_of(dflt_top_name)],
    )
    # para.h
    codes["para.h"] += [
        "#ifndef __PARA_H__\n",
        "#define __PARA_H__\n",
        "\n",
        "// COMMON\n",
    ]
    # common_para
    codes["para.h"] += para_code_block(common_para)
    # weight.h
    codes["weight.h"] += [
        "#ifndef __WEIGHT_H__\n",
        "#define __WEIGHT_H__\n",
    ]


def append_perlayer_code(codes, layer, prev_name, first_layer=False, last_layer=False):
    """append perlayer code to codes, including fifo, comp"""
    name = layer["name"]
    # para.h comment
    codes["para.h"] += ["\n", f"// {legal_up(name)}\n"]
    # weight.h comment
    codes["weight.h"] += ["\n", f"// {legal_up(name)}\n"]
    # define layer afifo
    codes["fifo"] += fifo_code_block(name)
    # layer comp
    common_tpl_list = [
        "CFG_AW",
        "CFG_WW",
        # "CFG_PW",
        # "CFG_SW",
        # "CFG_BW",
        # "CFG_MW",
        "CFG_EXP",
    ]
    common_arg_list = [
        afifo_of(prev_name),
        afifo_of(name),
        tfifo_of(prev_name),
        tfifo_of(name),
    ]
    if layer["type"] == "conv":
        def_k_list = [cfg_of(name, k) for k in ["PIC", "POC", "IC", "OC", "H", "W"]]
        def_v_list = layer["parallelism"] + layer["channels"] + layer["input_shape"]
        def_k_list += [cfg_of(name, f"{k}W") for k in ["P", "S", "B"]]
        def_v_list += [
            f"(CFG_AW + CFG_WW + {math.ceil(math.log2(layer['channels'][0]))})",
            "CFG_SW",
            "CFG_BW",
        ]
        if layer["name"] == "conv1":
            assert first_layer
            # add additional comp layer
            func_name = "conv_3x3_first_layer"
            unique_tpl_list = def_k_list.copy()
            unique_tpl_list.remove(cfg_of(name, "IC"))
            unique_arg_list = [ts2fifo_of(dflt_top_name), wbuf_of(name), sbuf_of(name)]
            # weight.h wbuf
            codes["weight.h"] += variable_code_block(
                name,
                "w",
                f"{cfg_of(name, 'PIC')}*CFG_WW",
                f"9][{cfg_of(name, 'OC')}",
            )
            # weight.h sbuf
            codes["weight.h"] += variable_code_block(
                name,
                "s",
                f"{cfg_of(name, 'SW')}+{cfg_of(name, 'BW')}",
                f"{cfg_of(name, 'OC')}",
            )
        elif layer["name"] == "conv8":
            func_name = "conv8"
            unique_tpl_list = def_k_list.copy()
            unique_arg_list = [wbuf_of(name), sbuf_of(name)]
            # weight.h wbuf
            codes["weight.h"] += variable_code_block(
                name,
                "w",
                f"{cfg_of(name, 'PIC')}*CFG_WW",
                f"{cfg_of(name, 'OC')}][{cfg_of(name, 'IC')}/{cfg_of(name, 'PIC')}",
            )
            # weight.h sbuf
            codes["weight.h"] += variable_code_block(
                name,
                "s",
                f"{cfg_of(name, 'SW')}+{cfg_of(name, 'BW')}",
                f"{cfg_of(name, 'OC')}",
            )
        else:
            raise ValueError(f"unknown kernel size {layer['kernel_size']}")
    elif layer["type"] == "block":
        assert layer["stride"] in [1, 2]
        assert not (layer["stride"] == 2 and layer["residual"])
        def_k_list = [
            cfg_of(name, k) for k in ["PIC", "PC", "POC", "IC", "C", "OC", "H", "W"]
        ]
        def_v_list = layer["parallelism"] + layer["channels"] + layer["input_shape"]
        def_k_list += [
            cfg_of(name, f"{k}W{i}") for k in ["P", "S", "B"] for i in range(3)
        ]
        def_v_list += [
            f"(CFG_AW + CFG_WW + {math.ceil(math.log2(layer['channels'][0]))})",
            f"(CFG_AW + CFG_WW + 4)",
            f"(CFG_AW + CFG_WW + {math.ceil(math.log2(layer['channels'][2]))})",
        ]
        def_v_list += ["CFG_SW"] * 3
        def_v_list += ["CFG_BW"] * 3
        func_name = (
            "conv_1x1_3x3_dw_1x1"
            + f"_stride{layer['stride']}"
            + ("_residual" if layer["residual"] else "")
        )
        if layer["residual"]:
            def_k_list.append(cfg_of(name, "IW"))
            def_v_list.append("(CFG_SW + 1)")
        # unique_tpl_list = def_k_list[3:6] + def_k_list[:3] + def_k_list[6:]
        unique_tpl_list = deepcopy(def_k_list)
        if layer["residual"]:
            unique_tpl_list.remove(cfg_of(name, "OC"))
        unique_arg_list = [wbuf_of(f"{name}_{i}") for i in range(3)] + [
            sbuf_of(f"{name}_{i}") for i in range(3)
        ]
        if layer["residual"]:
            unique_arg_list += [ibuf_of(name)]
        # wbuf0
        codes["weight.h"] += variable_code_block(
            f"{name}_0",
            "w",
            f"{cfg_of(name, 'PIC')}*CFG_WW",
            f"{cfg_of(name, 'C')}][{cfg_of(name, 'IC')}/{cfg_of(name, 'PIC')}",
        )
        # wbuf1
        codes["weight.h"] += variable_code_block(
            f"{name}_1",
            "w",
            f"{cfg_of(name, 'PC')}*CFG_WW",
            f"9][{cfg_of(name, 'C')}/{cfg_of(name, 'PC')}",
        )
        # wbuf2
        codes["weight.h"] += variable_code_block(
            f"{name}_2",
            "w",
            f"{cfg_of(name, 'PC')}*CFG_WW",
            f"{cfg_of(name, 'OC')}][{cfg_of(name, 'C')}/{cfg_of(name, 'PC')}",
        )
        # sbuf0
        codes["weight.h"] += variable_code_block(
            f"{name}_0",
            "s",
            f"{cfg_of(name, 'SW0')}+{cfg_of(name, 'BW0')}",
            f"{cfg_of(name, 'C')}",
        )
        # sbuf1
        codes["weight.h"] += variable_code_block(
            f"{name}_1",
            "s",
            f"{cfg_of(name, 'SW1')}+{cfg_of(name, 'BW1')}",
            f"{cfg_of(name, 'C')}",
        )
        # sbuf2
        codes["weight.h"] += variable_code_block(
            f"{name}_2",
            "s",
            f"{cfg_of(name, 'SW2')}+{cfg_of(name, 'BW2')}",
            f"{cfg_of(name, 'OC')}",
        )
        # ibuf
        if layer["residual"]:
            codes["weight.h"] += variable_code_block(
                name,
                "i",
                f"{cfg_of(name, 'IW')}",
                None,
            )
    elif layer["type"] == "linear":
        func_name = "global_avgpool_linear"
        def_k_list = [cfg_of(name, k) for k in ["PIC", "POC", "IC", "OC", "H", "W"]]
        def_v_list = layer["parallelism"] + layer["channels"] + layer["input_shape"]
        unique_tpl_list = def_k_list.copy()
        common_arg_list = []  # clear common_arg_list
        unique_arg_list = [
            afifo_of(prev_name),
            tfifo_of(prev_name),
            "act_out",
            wbuf_of(name),
        ]
        # weight.h wbuf
        codes["weight.h"] += variable_code_block(
            name,
            "w",
            f"{cfg_of(name, 'PIC')}*CFG_WW",
            f"{cfg_of(name, 'OC')}][{cfg_of(name, 'IC')}/{cfg_of(name, 'PIC')}",
        )
    else:
        raise ValueError(f"unknown layer type {layer['type']}")
    tpl_list = unique_tpl_list + common_tpl_list
    arg_list = common_arg_list + unique_arg_list
    codes["comp"] += func_code_block(func_name, tpl_list, arg_list)
    # para.h
    assert len(def_k_list) == len(def_v_list), f"{def_k_list}\n{def_v_list}"
    def_dict = {k: v for k, v in zip(def_k_list, def_v_list)}
    codes["para.h"] += para_code_block(def_dict)


def append_epilouge_code(codes, top_para, prev_name):
    """append epilouge code to codes, including store"""
    # store
    # codes["store"] += func_code_block(
    #     "write_output",
    #     [cfg_of(dflt_top_name, k) for k in ["POC", "OC"]]
    #     + ["CFG_AW"]
    #     + [cfg_of(dflt_top_name, k) for k in ["OH", "OW"]],
    #     [afifo_of(prev_name), tfifo_of(prev_name), "act_out"],
    # )
    # para.h
    codes["para.h"] += ["\n", "// TOP\n"]
    codes["para.h"] += para_code_block(top_para)
    codes["para.h"] += ["\n", "#endif\n"]
    # weight.h
    codes["weight.h"] += ["\n", "#endif\n"]


def gen_code(top_tpl, cfg, work_dir):
    top_tags = ["fifo", "load", "comp", "store"]
    header_tags = ["para.h", "weight.h"]
    tags = top_tags + header_tags
    codes = {tag: [] for tag in tags}
    # preprocess cfg
    if cfg["dataset"] == "NCAL":
        cfg["CFG_SW"] = 32
        cfg["CFG_BW"] = 32
        cfg["CFG_EXP"] = 32
    if cfg["dataset"] == "Roshambo":
        cfg["CFG_MW"] = 64
    # init common_para
    common_para = {
        "CFG_AW": "8   // AW",
        "CFG_WW": "8   // WW",
        "CFG_PW": "32  // PSUMW",
        "CFG_SW": f"{cfg.get('CFG_SW', 32) }  // SCALEW",
        "CFG_BW": f"{cfg.get('CFG_BW', 32) }  // BIASW",
        "CFG_TW": "8   // TOKENW",
        "CFG_MW": f"{cfg.get('CFG_MW', 128) }  // to be very carefull",
        "CFG_EXP": f"{cfg.get('CFG_EXP', 32) } // EXP",
    }
    # init top_para
    top_para = {}
    name = cfg["layers"][0]["name"]
    top_para.update({cfg_of(dflt_top_name, k): cfg_of(name, k) for k in ["IC", "PIC"]})
    top_para.update(
        {cfg_of(dflt_top_name, f"I{k}"): cfg_of(name, k) for k in ["H", "W"]}
    )
    name = cfg["layers"][-1]["name"]
    stride = cfg["layers"][-1]["stride"]
    top_para.update({cfg_of(dflt_top_name, k): cfg_of(name, k) for k in ["OC", "POC"]})
    top_para.update(
        {
            cfg_of(dflt_top_name, f"O{k}"): f"({cfg_of(name, k)}/{stride})"
            for k in ["H", "W"]
        }
    )

    # prolouge
    append_prolouge_code(codes, common_para)
    # per layer
    prev_name = dflt_top_name
    for idx, layer in enumerate(cfg["layers"]):
        append_perlayer_code(
            codes,
            layer,
            prev_name,
            first_layer=(idx == 0),
            last_layer=(idx == len(cfg["layers"]) - 1),
        )
        # update prev_name
        prev_name = layer["name"]
    # epilouge
    append_epilouge_code(codes, top_para, prev_name)
    # insert codes to top_out and write to file
    top_out = deepcopy(top_tpl)
    for tag in top_tags:
        try:
            pos = top_out.index(f"    /*gen_code-{tag}*/\n")
            top_out = top_out[: pos + 1] + codes[tag] + top_out[pos + 1 :]
        except ValueError:
            raise ValueError(f"gen_code-{tag} not found in top_tpl")
    open(os.path.join(work_dir, "top.cpp"), "w").writelines(top_out)
    print("top.cpp generated.")
    # write header by tag
    for tag in header_tags:
        header_lines = deepcopy(codes[tag])
        open(os.path.join(work_dir, tag), "w").writelines(header_lines)
        print(f"{tag} generated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", "-d", type=str, default=".")
    parser.add_argument("--force", "-f", action="store_true", default=False)
    args = parser.parse_args()

    # check work_dir
    if not os.path.exists(args.work_dir):
        raise FileNotFoundError(f"work_dir {args.work_dir} not found.")

    # check read files
    for filename in ["cfg.json", "top.cpp.tpl"]:
        if not os.path.exists(os.path.join(args.work_dir, filename)):
            raise FileNotFoundError(f"{filename} not found.")

    # check write files
    for filename in ["top.cpp", "para.h", "weight.h"]:
        if not args.force and os.path.exists(os.path.join(args.work_dir, filename)):
            raise FileExistsError(f"{filename} already exists. Use -f to overwrite.")

    # read top_tpl and cfg
    top_tpl = open(os.path.join(args.work_dir, "top.cpp.tpl")).readlines()
    cfg = json.load(open(os.path.join(args.work_dir, "cfg.json")))

    # generate code
    assert cfg["dataset"] in valid_dataset_list
    gen_code(top_tpl, cfg, args.work_dir)


if __name__ == "__main__":
    main()
