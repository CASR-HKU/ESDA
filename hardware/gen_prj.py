from copy import deepcopy
import math
import shutil
import os
import sys
import argparse
import json
import time
from template.common import *


def po2_p(n: int, x: int) -> int:
    """Floor n to the nearest power of 2 and also a factor of x"""
    if n > 1:
        power_of_2 = 1 << (n.bit_length() - 1)
        factor = math.gcd(power_of_2, x)
        print(f"Floor {n} to {factor} (factor of {x})")
        return factor
    else:
        raise ValueError("n must be greater than 1")


def get_bare_cfg(args) -> dict:
    cfg = args.cfg
    new_cfg = {k: cfg[k] for k in cfg if k not in ["layers"]}
    new_cfg["CFG_SW"] = 16
    new_cfg["CFG_BW"] = 16
    new_cfg["CFG_EXP"] = 16
    return new_cfg


def copy_dir(src_dir, dst_dir, cfg, args):
    copy_mode = "fresh"
    if os.path.exists(dst_dir):
        if args.overwrite:
            copy_mode = "overwrite"
        elif args.force:
            shutil.rmtree(dst_dir)
            copy_mode = "force"
        else:
            print(f"Skip {src_dir} to {dst_dir} (exists)")
            return
    if not args.dryrun:
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        if args.po2_axi:
            cfg["layers"][0]["parallelism"][0] = po2_p(
                cfg["layers"][0]["parallelism"][0], cfg["layers"][0]["channels"][0]
            )
            cfg["layers"][-1]["parallelism"][-1] = po2_p(
                cfg["layers"][-1]["parallelism"][-1], cfg["layers"][-1]["channels"][-1]
            )
        json.dump(cfg, open(f"{dst_dir}/cfg.json", "w"), indent=4)
    print(f"Copy {src_dir} to {dst_dir} ({copy_mode})")


def gen_single_blk(args):
    bare_cfg = get_bare_cfg(args)
    # loop all layers
    for blk in args.cfg["layers"]:
        bare_cfg["layers"] = deepcopy([blk])
        prj_name = legal_low(blk["name"])
        dst_dir = os.path.join(args.dst_dir, f"single_blk-{prj_name}")
        copy_dir(args.tpl_dir, dst_dir, bare_cfg, args)


def gen_multi_blk(args):
    bare_cfg = get_bare_cfg(args)
    # gen multi_blk
    gen_len = 6
    for beg_idx in range(0, len(bare_cfg["layers"]), gen_len):
        end_idx = min(beg_idx + gen_len, len(bare_cfg["layers"]))
        bare_cfg["layers"] = deepcopy(args.cfg["layers"][beg_idx:end_idx])
        prj_name = (
            f"{legal_low(bare_cfg['layers'][beg_idx]['name'])}"
            + f"-{legal_low(bare_cfg['layers'][end_idx-1]['name'])}"
        )
        dst_dir = os.path.join(args.dst_dir, f"multi_blk-{prj_name}")
        copy_dir(args.tpl_dir, dst_dir, bare_cfg, args)


def gen_full(args):
    bare_cfg = get_bare_cfg(args)
    bare_cfg["layers"] = deepcopy(args.cfg["layers"])
    dst_dir = os.path.join(args.dst_dir, "full")
    copy_dir(args.tpl_dir, dst_dir, bare_cfg, args)


def main():
    # save command to gen_prj.log with timestamp
    open("gen_prj.log", "a").write(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(sys.argv)}\n"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("--cfg_name", type=str, default="default")
    parser.add_argument(
        "--cfg_path", default="/vol/datastore/eventNetConfig/default_runs"
    )
    parser.add_argument("--tpl_dir", default="./template")
    parser.add_argument("--dst_path", default="./debug")
    parser.add_argument("--po2_axi", action="store_true", default=False)
    parser.add_argument("--dryrun", "-d", action="store_true", default=False)
    parser.add_argument("--overwrite", "-o", action="store_true", default=False)
    parser.add_argument("--force", "-f", action="store_true", default=False)
    args = parser.parse_args()

    args.cfg = json.load(
        open(os.path.join(args.cfg_path, args.cfg_name, "en-result.json"))
    )
    args.dst_dir = os.path.join(args.dst_path, args.cfg_name)
    args.SBE16 = True

    if args.task not in ["gen_single_blk", "gen_multi_blk", "gen_full"]:
        raise NotImplementedError(f"Task {args.task} not implemented")

    if not args.dryrun:
        os.makedirs(args.dst_dir, exist_ok=True)
        shutil.copy(
            f"/vol/datastore/yuhao/event_spconv_auto_test/sub.mk",
            f"{args.dst_dir}/Makefile",
        )

    globals()[args.task](args)


if __name__ == "__main__":
    main()
