from copy import deepcopy
from typing import Any, Dict, List
from formulation.base_formulation import BaseFormulation
from utils.common import my_prod

# from utils.common import my_sum
from utils.dse_var import DSEVar
from utils.dse_constr import DSEConstr
from utils.simp_param import SimpParam

wdp_scale_list = [16, 1024, 1]  # 16x1024
wdp_bname_list = ["w0", "s0", "w1", "s1", "l1", "w2", "s2"]
wdp_conv1_bname_list = ["s", "l"]
wdp_conv8_bname_list = ["w", "s"]
wdp_vname_list = ["w_d16", "d_d1024", "p"]
wdp_var_w_1x1_func = lambda vd, name, pic, poc, tic, toc: [
    8 * vd[f"{pic}.{name}"],  # width
    vd[f"{tic}.{name}"] * vd[f"{toc}.{name}"] * 2,  # depth
    vd[f"{poc}.{name}"] / 2,  # partition
]
wdp_var_w_3x3_func = lambda vd, name, pic, poc, tic, toc: [
    8 * vd[f"{pic}.{name}"],  # width
    vd[f"{tic}.{name}"] * vd[f"{toc}.{name}"] * 2,  # depth
    9 * vd[f"{poc}.{name}"] / 2,  # partition
]
wdp_var_w_3x3_dw_func = lambda vd, name, pic, tic: [
    8 * vd[f"{pic}.{name}"],  # width
    9 * vd[f"{tic}.{name}"],  # depth
    1,  # partition
]
wdp_var_s_func = lambda vd, name, pic, tic: [
    32,  # width
    vd[f"{tic}.{name}"] * 2,  # depth
    vd[f"{pic}.{name}"] / 2,  # partition
]
wdp_var_l_func = lambda vd, name, pic, iw, tic: [
    8 * vd[f"{pic}.{name}"],  # width
    3 * iw * vd[f"{tic}.{name}"],  # depth
    1,  # partition
]
wdp_var_conv1_l_func = lambda vd, name, pic, iw, tic: [
    8 * vd[f"{pic}.{name}"],  # width
    iw * vd[f"{tic}.{name}"],  # depth
    3,  # partition
]
wdp_var_func_dict = {}
wdp_var_func_dict["w0"] = lambda vd, name: wdp_var_w_1x1_func(
    vd, name, "pic", "pc", "tic", "tc"
)
wdp_var_func_dict["s0"] = lambda vd, name: wdp_var_s_func(vd, name, "pc", "tc")
wdp_var_func_dict["w1"] = lambda vd, name: wdp_var_w_3x3_dw_func(vd, name, "pc", "tc")
wdp_var_func_dict["s1"] = lambda vd, name: wdp_var_s_func(vd, name, "pc", "tc")
wdp_var_func_dict["l1"] = lambda vd, name, iw: wdp_var_l_func(vd, name, "pc", iw, "tc")
wdp_var_func_dict["w2"] = lambda vd, name: wdp_var_w_1x1_func(
    vd, name, "pc", "poc", "tc", "toc"
)
wdp_var_func_dict["s2"] = lambda vd, name: wdp_var_s_func(vd, name, "poc", "toc")
# for conv1
wdp_var_conv1_func_dict = {}
# wdp_var_conv1_func_dict["w"] = lambda vd, name: wdp_var_w_3x3_func(
#     vd, name, "pic", "poc", "tic", "toc"
# )
wdp_var_conv1_func_dict["s"] = lambda vd, name: wdp_var_s_func(vd, name, "poc", "toc")
wdp_var_conv1_func_dict["l"] = lambda vd, name, iw: wdp_var_conv1_l_func(
    vd, name, "poc", iw, "toc"
)
# for conv8
wdp_var_conv8_func_dict = {}
wdp_var_conv8_func_dict["w"] = lambda vd, name: wdp_var_w_1x1_func(
    vd, name, "pic", "poc", "tic", "toc"
)
wdp_var_conv8_func_dict["s"] = lambda vd, name: wdp_var_s_func(vd, name, "poc", "toc")


class EventNetBlockFormulation(BaseFormulation):
    """Formulation by block (1x1 -> 3x3 dw -> 1x1)"""

    model: Dict[str, Any]
    hw: Dict[str, Any]

    result: Dict[str, Any]  # self defined result

    def __init__(
        self,
        name: str,
        path: str,
        model: Dict[str, Any],
        hw: Dict[str, Any],
        config: Dict[str, Any],
    ):
        self.model = model
        self.hw = hw
        config.setdefault("form-obj", "lat_max")  # lat_max, lat_max+uti, lat_sum
        config.setdefault("form-8b_pack", True)
        config.setdefault("form-ind_dsp", True)
        config.setdefault("form-match_io", True)
        config.setdefault("form-even_p", True)
        config.setdefault("form-fix_first_pic", False)
        config.setdefault("form-fix_all_p", None)
        super().__init__(name, path, config)

    def build_var_list(self) -> None:
        var = DSEVar("obj", 1, is_obj=True, vtype="CONTINUOUS")
        self.var_list.append(var)
        if self.config["form-obj"].startswith("lat_max"):
            var = DSEVar("lat_max", 1)
            self.var_list.append(var)
        var = DSEVar("dsp", 1)
        self.var_list.append(var)
        var = DSEVar("dsp_uti", ub=1, vtype="CONTINUOUS")
        self.var_list.append(var)
        var = DSEVar("bram", 1)
        self.var_list.append(var)
        var = DSEVar("bram_uti", ub=1, vtype="CONTINUOUS")
        self.var_list.append(var)
        for layer in self.model["layers"]:
            if layer["type"] == "conv":
                var = DSEVar(f"lat.{layer['name']}", 1)
                self.var_list.append(var)
                var = DSEVar(f"tic.{layer['name']}", 1)
                self.var_list.append(var)
                var = DSEVar(f"toc.{layer['name']}", 1)
                self.var_list.append(var)
                var = DSEVar(f"tk.{layer['name']}", 1, vtype="CONTINUOUS")
                self.var_list.append(var)
                var = DSEVar(f"pic.{layer['name']}", 1, is_strict=True)
                self.var_list.append(var)
                var = DSEVar(f"poc.{layer['name']}", 1, is_strict=True)
                self.var_list.append(var)
                var = DSEVar(f"pk.{layer['name']}", 1, is_strict=True)
                self.var_list.append(var)
                if self.config["form-even_p"]:
                    var = DSEVar(f"poc_d2.{layer['name']}")
                    self.var_list.append(var)
                if layer["name"] == "conv1":
                    buf_list = wdp_conv1_bname_list
                elif layer["name"] == "conv8":
                    buf_list = wdp_conv8_bname_list
                for buf in buf_list:
                    for vname in wdp_vname_list:
                        var = DSEVar(f"{buf}_{vname}.{layer['name']}", 1)
                        self.var_list.append(var)
            elif layer["type"] == "block":
                var = DSEVar(f"lat.{layer['name']}.conv0", 1)
                self.var_list.append(var)
                var = DSEVar(f"lat.{layer['name']}.conv1", 1)
                self.var_list.append(var)
                var = DSEVar(f"lat.{layer['name']}.conv2", 1)
                self.var_list.append(var)
                var = DSEVar(f"tic.{layer['name']}", 1)
                self.var_list.append(var)
                var = DSEVar(f"tc.{layer['name']}", 1)
                self.var_list.append(var)
                var = DSEVar(f"toc.{layer['name']}", 1)
                self.var_list.append(var)
                var = DSEVar(f"tk.{layer['name']}", 1, vtype="CONTINUOUS")
                self.var_list.append(var)
                var = DSEVar(f"pic.{layer['name']}", 1, is_strict=True)
                self.var_list.append(var)
                var = DSEVar(f"pc.{layer['name']}", 1, is_strict=True)
                self.var_list.append(var)
                var = DSEVar(f"poc.{layer['name']}", 1, is_strict=True)
                self.var_list.append(var)
                var = DSEVar(f"pk.{layer['name']}", 1, is_strict=True)
                self.var_list.append(var)
                if self.config["form-even_p"]:
                    var = DSEVar(f"pc_d2.{layer['name']}")
                    self.var_list.append(var)
                    var = DSEVar(f"poc_d2.{layer['name']}")
                    self.var_list.append(var)
                for x in range(3):
                    buf_list = ["w", "s"] if x != 1 else ["w", "s", "l"]
                    for buf in buf_list:
                        for vname in wdp_vname_list:
                            var = DSEVar(f"{buf}{x}_{vname}.{layer['name']}", 1)
                            self.var_list.append(var)
            else:
                raise ValueError("Unknown layer type")
            # common vars for each layers
            if self.config["form-ind_dsp"]:
                var = DSEVar(f"dsp.{layer['name']}", 1)
                self.var_list.append(var)
            var = DSEVar(f"bram.{layer['name']}", 1)
            self.var_list.append(var)

    def build_constr_list(self) -> None:
        match_io = self.config.get("form-match_io", False)
        self.build_constr_list_misc()
        for layer in self.model["layers"]:
            if layer["type"] == "conv" and layer["name"] == "conv1":
                self.build_constr_list_conv1(layer)
            elif layer["type"] == "conv" and layer["name"] == "conv8":
                self.build_constr_list_conv8(layer)
            elif layer["type"] == "block":
                self.build_constr_list_block_lat(layer)
                self.build_constr_list_block_hw(layer)
            else:
                raise ValueError("Unknown layer type")
        if match_io:
            self.build_constr_list_match_io()
        self.build_constr_list_hw()
        self.build_constr_list_obj()

    def build_constr_list_misc(self) -> None:
        # fix pic of first layer
        if self.config["form-fix_first_pic"]:
            layer = self.model["layers"][0]
            constr = DSEConstr(
                "c.pic.first_pic",
                lambda vd: vd[f"pic.{layer['name']}"],
                inter_var=layer["channels"][0],
            )
            self.constr_list.append(constr)
        # fix all p
        if self.config["form-fix_all_p"] is not None:
            p = self.config["form-fix_all_p"]
            for layer in self.model["layers"]:
                name = layer["name"]
                if layer["type"] == "conv":
                    k_list = ["pic", "poc"]
                elif layer["type"] == "block":
                    k_list = ["pic", "pc", "poc"]
                for k in k_list:
                    constr = DSEConstr(
                        f"c.fix_{k}.{name}",
                        lambda vd, name=name, k=k: vd[f"{k}.{name}"],
                        inter_var=p,
                    )
                    self.constr_list.append(constr)

    def build_constr_list_conv1(self, layer: dict) -> None:
        name = layer["name"]
        # lat_max >= sparsity * tk.i * lat.i
        if self.config["form-obj"].startswith("lat_max"):
            constr = DSEConstr(
                f"c.lat_max.{name}",
                lambda vd: vd["lat_max"]
                >= layer["sparsity"] * vd[f"tk.{name}"] * vd[f"lat.{name}"],
            )
            self.constr_list.append(constr)
        # lat.i == (H//stride) * (W//stride) * tic.i * toc.i
        constr = DSEConstr(
            f"c.lat.{name}",
            lambda vd: (layer["input_shape"][0] // layer["stride"])
            * (layer["input_shape"][1] // layer["stride"])
            * vd[f"tic.{name}"]
            * vd[f"toc.{name}"],
            inter_var=f"lat.{name}",
        )
        self.constr_list.append(constr)
        # tic.i * pic.i == IC
        constr = DSEConstr(
            f"c.tic.{name}",
            lambda vd: vd[f"tic.{name}"] * vd[f"pic.{name}"] == layer["channels"][0],
        )
        self.constr_list.append(constr)
        # toc.i * poc.i == OC
        constr = DSEConstr(
            f"c.toc.{name}",
            lambda vd: vd[f"toc.{name}"] * vd[f"poc.{name}"] == layer["channels"][1],
        )
        self.constr_list.append(constr)
        # tk.i == 1, pk.i == 9
        constr = DSEConstr(f"c.tk.{name}", lambda vd: vd[f"tk.{name}"], 1)
        self.constr_list.append(constr)
        constr = DSEConstr(f"c.pk.{name}", lambda vd: vd[f"pk.{name}"], 9)
        self.constr_list.append(constr)
        # even_p
        if self.config["form-even_p"]:
            constr = DSEConstr(
                f"c.even_poc.{name}", lambda vd: vd[f"poc_d2.{name}"] * 2, f"poc.{name}"
            )
            self.constr_list.append(constr)
        # dsp
        if self.config["form-ind_dsp"]:
            if self.config["form-8b_pack"]:
                # dsp.i >= pic.i * poc.i * pk.i / 2 + poc.i
                dsp_func = lambda vd: (
                    vd[f"pic.{name}"] * vd[f"poc.{name}"] * vd[f"pk.{name}"] / 2
                    + vd[f"poc.{name}"]
                )
            else:
                raise NotImplementedError("dsp function for 16b not implemented")
        constr = DSEConstr(
            f"c.dsp.{name}.gpkit",
            lambda vd: dsp_func(vd) <= vd[f"dsp.{name}"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        # dsp_func(vd) == dsp.i (scip)
        constr = DSEConstr(
            f"c.dsp.{name}.scip",
            dsp_func,
            inter_var=f"dsp.{name}",
            scope="scip",
        )
        self.constr_list.append(constr)
        # <bname>_<vname>.i
        for bname in wdp_conv1_bname_list:
            # wrap as wrapped_func(vd) -> [w, d, p]
            if bname == "l":
                iw = layer["input_shape"][0]
                wrapped_func = lambda vd, name=name, iw=iw: wdp_var_conv1_func_dict[
                    bname
                ](vd, name, iw)
            else:
                wrapped_func = (
                    lambda vd, bname=bname, name=name: wdp_var_conv1_func_dict[bname](
                        vd, name
                    )
                )
            # w, d, p
            for vidx, vname in enumerate(wdp_vname_list):
                bvname = f"{bname}_{vname}.{name}"
                constr = DSEConstr(
                    f"c.{bvname}",
                    lambda vd, f=wrapped_func, vidx=vidx, bvname=bvname: f(vd)[vidx]
                    <= vd[f"{bvname}"] * wdp_scale_list[vidx],
                )
                self.constr_list.append(constr)
        # bram.i
        # sum(w*d*p) <= bram.i (gpkit)
        constr = DSEConstr(
            f"c.bram.{name}.gpkit",
            lambda vd: sum(
                my_prod(vd[f"{bname}_{vname}.{name}"] for vname in wdp_vname_list)
                for bname in wdp_conv1_bname_list
            )
            <= vd[f"bram.{name}"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        # sum(w*d*p) == bram.i (scip)
        constr = DSEConstr(
            f"c.bram.{name}.scip",
            lambda vd: sum(
                my_prod(vd[f"{bname}_{vname}.{name}"] for vname in wdp_vname_list)
                for bname in wdp_conv1_bname_list
            ),
            inter_var=f"bram.{name}",
            scope="scip",
        )
        self.constr_list.append(constr)

    def build_constr_list_conv8(self, layer: dict) -> None:
        name = layer["name"]
        # lat_max >= sparsity * tk.i * lat.i
        if self.config["form-obj"].startswith("lat_max"):
            constr = DSEConstr(
                f"c.lat_max.{name}",
                lambda vd: vd["lat_max"]
                >= layer["sparsity"] * vd[f"tk.{name}"] * vd[f"lat.{name}"],
            )
            self.constr_list.append(constr)
        # lat.i == (H//stride) * (W//stride) * tic.i * toc.i
        constr = DSEConstr(
            f"c.lat.{name}",
            lambda vd: (layer["input_shape"][0] // layer["stride"])
            * (layer["input_shape"][1] // layer["stride"])
            * vd[f"tic.{name}"]
            * vd[f"toc.{name}"],
            inter_var=f"lat.{name}",
        )
        self.constr_list.append(constr)
        # tic.i * pic.i == IC
        constr = DSEConstr(
            f"c.tic.{name}",
            lambda vd: vd[f"tic.{name}"] * vd[f"pic.{name}"] == layer["channels"][0],
        )
        self.constr_list.append(constr)
        # toc.i * poc.i == OC
        constr = DSEConstr(
            f"c.toc.{name}",
            lambda vd: vd[f"toc.{name}"] * vd[f"poc.{name}"] == layer["channels"][1],
        )
        self.constr_list.append(constr)
        # tk.i == 1, pk.i == 1
        constr = DSEConstr(f"c.tk.{name}", lambda vd: vd[f"tk.{name}"], 1)
        self.constr_list.append(constr)
        constr = DSEConstr(f"c.pk.{name}", lambda vd: vd[f"pk.{name}"], 1)
        self.constr_list.append(constr)
        # even_p
        if self.config["form-even_p"]:
            constr = DSEConstr(
                f"c.even_poc.{name}", lambda vd: vd[f"poc_d2.{name}"] * 2, f"poc.{name}"
            )
            self.constr_list.append(constr)
        # dsp
        if self.config["form-ind_dsp"]:
            if self.config["form-8b_pack"]:
                # dsp.i >= pic.i * poc.i * pk.i / 2 + poc.i
                dsp_func = lambda vd: (
                    vd[f"pic.{name}"] * vd[f"poc.{name}"] * vd[f"pk.{name}"] / 2
                    + vd[f"poc.{name}"]
                )
            else:
                raise NotImplementedError("dsp function for 16b not implemented")
        constr = DSEConstr(
            f"c.dsp.{name}.gpkit",
            lambda vd: dsp_func(vd) <= vd[f"dsp.{name}"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        # dsp_func(vd) == dsp.i (scip)
        constr = DSEConstr(
            f"c.dsp.{name}.scip",
            dsp_func,
            inter_var=f"dsp.{name}",
            scope="scip",
        )
        self.constr_list.append(constr)
        # <bname>_<vname>.i
        for bname in wdp_conv8_bname_list:
            # wrap as wrapped_func(vd) -> [w, d, p]
            wrapped_func = lambda vd, bname=bname, name=name: wdp_var_conv8_func_dict[
                bname
            ](vd, name)
            # w, d, p
            for vidx, vname in enumerate(wdp_vname_list):
                bvname = f"{bname}_{vname}.{name}"
                constr = DSEConstr(
                    f"c.{bvname}",
                    lambda vd, f=wrapped_func, vidx=vidx, bvname=bvname: f(vd)[vidx]
                    <= vd[f"{bvname}"] * wdp_scale_list[vidx],
                )
                self.constr_list.append(constr)
        # bram.i
        # sum(w*d*p) <= bram.i (gpkit)
        constr = DSEConstr(
            f"c.bram.{name}.gpkit",
            lambda vd: sum(
                my_prod(vd[f"{bname}_{vname}.{name}"] for vname in wdp_vname_list)
                for bname in wdp_conv8_bname_list
            )
            <= vd[f"bram.{name}"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        # sum(w*d*p) == bram.i (scip)
        constr = DSEConstr(
            f"c.bram.{name}.scip",
            lambda vd: sum(
                my_prod(vd[f"{bname}_{vname}.{name}"] for vname in wdp_vname_list)
                for bname in wdp_conv8_bname_list
            ),
            inter_var=f"bram.{name}",
            scope="scip",
        )
        self.constr_list.append(constr)

    def build_constr_list_block_lat(self, layer: dict) -> None:
        name = layer["name"]
        # lat_max >= sparsity * lat.i.convx, x = 0, 1, 2
        if self.config["form-obj"].startswith("lat_max"):
            constr = DSEConstr(
                f"c.lat_max.{name}.conv0",
                lambda vd: vd["lat_max"]
                >= layer["sparsity"][0] * vd[f"lat.{name}.conv0"],
            )
            self.constr_list.append(constr)
            constr = DSEConstr(
                f"c.lat_max.{name}.conv1",
                lambda vd: vd["lat_max"]
                >= layer["sparsity"][1] * vd[f"tk.{name}"] * vd[f"lat.{name}.conv1"],
            )
            self.constr_list.append(constr)
            constr = DSEConstr(
                f"c.lat_max.{name}.conv2",
                lambda vd: vd["lat_max"]
                >= layer["sparsity"][2] * vd[f"lat.{name}.conv2"],
            )
            self.constr_list.append(constr)
        # lat.i.conv0 == H * W * tic.i * tc.i
        constr = DSEConstr(
            f"c.lat.{name}.conv0",
            lambda vd: layer["input_shape"][0]
            * layer["input_shape"][1]
            * vd[f"tic.{name}"]
            * vd[f"tc.{name}"],
            inter_var=f"lat.{name}.conv0",
        )
        self.constr_list.append(constr)
        # lat.i.conv1 == (H//stride) * (W//stride) * tc.i * tk.i
        constr = DSEConstr(
            f"c.lat.{name}.conv1",
            lambda vd: (layer["input_shape"][0] // layer["stride"])
            * (layer["input_shape"][1] // layer["stride"])
            * vd[f"tc.{name}"],
            inter_var=f"lat.{name}.conv1",
        )
        self.constr_list.append(constr)
        # lat.i.conv2 == (H//stride) * (W//stride) * tc.i * toc.i
        constr = DSEConstr(
            f"c.lat.{name}.conv2",
            lambda vd: (layer["input_shape"][0] // layer["stride"])
            * (layer["input_shape"][1] // layer["stride"])
            * vd[f"tc.{name}"]
            * vd[f"toc.{name}"],
            inter_var=f"lat.{name}.conv2",
        )
        self.constr_list.append(constr)
        # tx.i * px.i == C[x], x = 0, 1, 2
        for x_idx, x in enumerate(["ic", "c", "oc"]):
            constr = DSEConstr(
                f"c.t{x}.{name}",
                lambda vd, x=x: vd[f"t{x}.{name}"] * vd[f"p{x}.{name}"],
                inter_var=layer["channels"][x_idx],
            )
            self.constr_list.append(constr)
        # tk.i == sum(kernel_sparsity[x]*(x+1) for x in range(9))
        constr = DSEConstr(
            f"c.tk.{name}",
            lambda vd: vd[f"tk.{name}"],
            inter_var=sum(layer["kernel_sparsity"][x] * (x + 1) for x in range(9)),
        )
        self.constr_list.append(constr)
        # pk.i == 1
        constr = DSEConstr(f"c.pk.{name}", lambda vd: vd[f"pk.{name}"], inter_var=1)
        self.constr_list.append(constr)
        # even_p
        if self.config["form-even_p"]:
            constr = DSEConstr(
                f"c.even_pc.{name}", lambda vd: vd[f"pc_d2.{name}"] * 2, f"pc.{name}"
            )
            self.constr_list.append(constr)
            constr = DSEConstr(
                f"c.even_poc.{name}", lambda vd: vd[f"poc_d2.{name}"] * 2, f"poc.{name}"
            )
            self.constr_list.append(constr)

    def build_constr_list_block_hw(self, layer: dict) -> None:
        name = layer["name"]
        # dsp.i
        if self.config["form-ind_dsp"]:
            if self.config["form-8b_pack"]:
                # dsp_func =
                # pic.i * pc.i / 2 + pc.i         (1x1 conv + quant)
                # + pc.i * pk.i + pc.i            (3x3 conv dw + quant)
                # + pc.i * poc.i / 2 + poc.i       (1x1 conv + quant)
                # + poc.i                           (residual quant)(optional)
                dsp_func = lambda vd: (
                    vd[f"pic.{name}"] * vd[f"pc.{name}"] / 2
                    + vd[f"pc.{name}"]
                    + vd[f"pc.{name}"] * vd[f"pk.{name}"]
                    + vd[f"pc.{name}"]
                    + vd[f"pc.{name}"] * vd[f"poc.{name}"] / 2
                    + (vd[f"poc.{name}"] if layer["residual"] else 0)
                )
            else:
                raise NotImplementedError("dsp function for 16b not implemented")
            # dsp_func(vd) <= dsp.i (gpkit)
            constr = DSEConstr(
                f"c.dsp.{name}.gpkit",
                lambda vd: dsp_func(vd) <= vd[f"dsp.{name}"],
                scope="gpkit",
            )
            self.constr_list.append(constr)
            # dsp_func(vd) == dsp.i (scip)
            constr = DSEConstr(
                f"c.dsp.{name}.scip",
                dsp_func,
                inter_var=f"dsp.{name}",
                scope="scip",
            )
            self.constr_list.append(constr)
        # <bname>_<vname>.i
        for bname in wdp_bname_list:
            # wrap as wrapped_func(vd) -> [w, d, p]
            if bname == "l1":
                iw = layer["input_shape"][0]
                wrapped_func = (
                    lambda vd, bname=bname, name=name, iw=iw: wdp_var_func_dict[bname](
                        vd, name, iw
                    )
                )
            else:
                wrapped_func = lambda vd, bname=bname, name=name: wdp_var_func_dict[
                    bname
                ](vd, name)
            # w, d, p
            for vidx, vname in enumerate(wdp_vname_list):
                bvname = f"{bname}_{vname}.{name}"
                constr = DSEConstr(
                    f"c.{bvname}",
                    lambda vd, f=wrapped_func, vidx=vidx, bvname=bvname: f(vd)[vidx]
                    <= vd[f"{bvname}"] * wdp_scale_list[vidx],
                )
                self.constr_list.append(constr)
        # bram.i
        # sum(w*d*p) <= bram.i (gpkit)
        constr = DSEConstr(
            f"c.bram.{name}.gpkit",
            lambda vd: sum(
                my_prod(vd[f"{bname}_{vname}.{name}"] for vname in wdp_vname_list)
                for bname in wdp_bname_list
            )
            <= vd[f"bram.{name}"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        # sum(w*d*p) == bram.i (scip)
        constr = DSEConstr(
            f"c.bram.{name}.scip",
            lambda vd: sum(
                my_prod(vd[f"{bname}_{vname}.{name}"] for vname in wdp_vname_list)
                for bname in wdp_bname_list
            ),
            inter_var=f"bram.{name}",
            scope="scip",
        )
        self.constr_list.append(constr)

    def build_constr_list_match_io(self) -> None:
        # match output of i with input of i+1
        for idx, layer in enumerate(self.model["layers"][:-1]):
            name = layer["name"]
            next_layer = self.model["layers"][idx + 1]
            next_name = next_layer["name"]
            # poc.i == pic.<i+1>
            constr = DSEConstr(
                f"c.io.{name}",
                lambda vd, k=f"poc.{name}": vd[k],
                inter_var=f"pic.{next_name}",
            )
            self.constr_list.append(constr)

    def build_constr_list_hw(self) -> None:
        name_list = [layer["name"] for layer in self.model["layers"]]
        if self.config["form-ind_dsp"]:
            # dsp >= sum(dsp.i) (gpkit)
            constr = DSEConstr(
                "c.dsp.gpkit",
                lambda vd: vd["dsp"] >= sum(vd[f"dsp.{name}"] for name in name_list),
                scope="gpkit",
            )
            self.constr_list.append(constr)
            # dsp == sum(dsp.i) (scip)
            constr = DSEConstr(
                "c.dsp.scip",
                lambda vd: sum(vd[f"dsp.{name}"] for name in name_list),
                inter_var="dsp",
                scope="scip",
            )
            self.constr_list.append(constr)
        else:
            constr = DSEConstr(
                "c.dsp",
                lambda vd: vd["dsp"]
                >= sum(
                    (
                        vd[f"pic.{layer['name']}"]
                        * vd[f"poc.{layer['name']}"]
                        * vd[f"pk.{layer['name']}"]
                    )
                    if layer["type"] == "conv"
                    else (
                        vd[f"pc0.{layer['name']}"] * vd[f"pc1.{layer['name']}"]
                        + vd[f"pc1.{layer['name']}"] * vd[f"pk.{layer['name']}"]
                        + vd[f"pc1.{layer['name']}"] * vd[f"pc2.{layer['name']}"]
                    )
                    for layer in self.model["layers"]
                ),
            )
            self.constr_list.append(constr)
            raise NotImplementedError("wrong formulation")
        # dsp / hw_dsp == dsp_uti
        constr = DSEConstr(
            "c.dsp_uti", lambda vd: vd["dsp"] / self.hw["dsp"], inter_var="dsp_uti"
        )
        self.constr_list.append(constr)
        # bram
        # bram >= sum(bram.i) (gpkit)
        constr = DSEConstr(
            "c.bram.gpkit",
            lambda vd: vd["bram"] >= sum(vd[f"bram.{name}"] for name in name_list),
            scope="gpkit",
        )
        self.constr_list.append(constr)
        # bram == sum(bram.i) (scip)
        constr = DSEConstr(
            "c.bram.scip",
            lambda vd: sum(vd[f"bram.{name}"] for name in name_list),
            inter_var="bram",
            scope="scip",
        )
        self.constr_list.append(constr)
        # bram / hw_bram == bram_uti
        constr = DSEConstr(
            "c.bram_uti",
            lambda vd: vd["bram"] / (self.hw["bram36"] * 2),
            inter_var="bram_uti",
        )
        self.constr_list.append(constr)

    def build_constr_list_obj(self) -> None:
        if self.config["form-obj"] == "lat_max":
            obj_func = lambda vd: vd["lat_max"]
        elif self.config["form-obj"] == "lat_max+uti":
            obj_func = (
                lambda vd: vd["lat_max"] + 0.5 * vd["dsp_uti"] + 0.5 * vd["bram_uti"]
            )
        elif self.config["form-obj"] == "lat_sum":
            obj_func = lambda vd: sum(
                vd[f"lat.{layer['name']}"]
                if layer["type"] == "conv"
                else (
                    vd[f"lat.{layer['name']}.conv1"]
                    + vd[f"lat.{layer['name']}.conv2"]
                    + vd[f"lat.{layer['name']}.conv3"]
                )
                for layer in self.model["layers"]
            )
            raise NotImplementedError("wrong formulation for lat_sum")
        else:
            raise ValueError(f"Unknown objective {self.config['form-obj']}")
        constr = DSEConstr(
            "c.obj.gpkit", lambda vd: vd["obj"] >= obj_func(vd), scope="gpkit"
        )
        self.constr_list.append(constr)
        constr = DSEConstr("c.obj.scip", obj_func, inter_var="obj", scope="scip")
        self.constr_list.append(constr)

    def build_result(self) -> None:
        if self.solution is not None:
            # copy all k:v from model to result except for k="layers"
            self.result = {k: v for k, v in self.model.items() if k != "layers"}
            # update all k:v from solution to result for k in ["obj", "dsp", "dsp_uti"]
            self.result.update(
                {
                    k: self.solution[k]
                    for k in ["obj", "dsp", "dsp_uti", "bram", "bram_uti"]
                }
            )
            # copy layers from model to result
            self.result["layers"] = deepcopy(self.model["layers"])
            for layer in self.result["layers"]:
                if layer["type"] == "conv":
                    layer["parallelism"] = [
                        self.solution[f"{k}.{layer['name']}"] for k in ["pic", "poc"]
                    ]
                    layer["lat"] = self.solution[f"lat.{layer['name']}"]
                    if layer["name"] == "conv1":
                        bname_list = wdp_conv1_bname_list
                    elif layer["name"] == "conv8":
                        bname_list = wdp_conv8_bname_list
                    for bname in bname_list:
                        layer[bname] = [
                            self.solution[f"{bname}_{vname}.{layer['name']}"]
                            for vname in wdp_vname_list
                        ]
                    if self.config["form-ind_dsp"]:
                        layer["dsp"] = self.solution[f"dsp.{layer['name']}"]
                    layer["bram"] = self.solution[f"bram.{layer['name']}"]
                elif layer["type"] == "block":
                    layer["parallelism"] = [
                        self.solution[f"{k}.{layer['name']}"]
                        for k in ["pic", "pc", "poc"]
                    ]
                    layer["lat"] = [
                        self.solution[f"lat.{layer['name']}.conv{x}"] for x in range(3)
                    ]
                    for bname in wdp_bname_list:
                        layer[bname] = [
                            self.solution[f"{bname}_{vname}.{layer['name']}"]
                            for vname in wdp_vname_list
                        ]
                    if self.config["form-ind_dsp"]:
                        layer["dsp"] = self.solution[f"dsp.{layer['name']}"]
                    layer["bram"] = self.solution[f"bram.{layer['name']}"]
                else:
                    raise ValueError(f"Unknown layer type {layer['type']}")
