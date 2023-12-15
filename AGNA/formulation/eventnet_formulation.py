from copy import deepcopy
import math
from typing import Any, Dict, List
from formulation.base_formulation import BaseFormulation
from utils.common import my_prod, my_sum
from utils.dse_var import DSEVar
from utils.dse_constr import DSEConstr
from utils.simp_param import SimpParam

top_lname = "top"

# lat_vfunc
conv_lat_vfunc = lambda vd, os, tic, toc: os[0] * os[1] * vd[tic] * vd[toc]
conv_dw_lat_vfunc = lambda vd, os, tic, toc: os[0] * os[1] * vd[tic]
linear_lat_vfunc = lambda vd, os, tic, toc: vd[tic] * vd[toc]
lat_vfunc_dict = {
    "conv_1x1": conv_lat_vfunc,
    "conv_3x3": conv_lat_vfunc,
    "conv_3x3_dw": conv_dw_lat_vfunc,
    "linear": linear_lat_vfunc,
}

# dsp_vfunc
conv_1x1_dsp_vfunc = lambda vd, pic, poc: vd[pic] * vd[poc] / 2 + vd[poc]
conv_3x3_dsp_vfunc = lambda vd, pic, poc: vd[pic] * vd[poc] * 9 / 2 + vd[poc]
conv_3x3_dw_dsp_vfunc = lambda vd, pic, poc: vd[pic] + vd[pic]
linear_dsp_vfunc = lambda vd, pic, poc: vd[pic] * vd[poc]
res_dsp_vfunc = lambda vd, pic, poc: vd[poc]
dsp_vfunc_dict = {
    "conv_1x1": conv_1x1_dsp_vfunc,
    "conv_3x3": conv_3x3_dsp_vfunc,
    "conv_3x3_dw": conv_3x3_dw_dsp_vfunc,
    "linear": linear_dsp_vfunc,
}

# buf_vfunc
buf_dscale_list = [16, 1024, 1]
buf_dname_list = ["w_dx", "d_dx", "p_dx"]
conv_1x1_wbuf_vfunc = {
    "w_dx": lambda vd, pic, **kw: 8 * vd[pic],
    "d_dx": lambda vd, tic, toc, **kw: vd[tic] * vd[toc] * 2,
    "p_dx": lambda vd, poc, **kw: vd[poc] / 2,
}
conv_3x3_wbuf_vfunc = {
    "w_dx": lambda vd, pic, **kw: 8 * vd[pic],
    "d_dx": lambda vd, tic, toc, **kw: 2 * vd[tic] * vd[toc],
    "p_dx": lambda vd, poc, **kw: 9 * vd[poc] / 2,
}
conv_3x3_dw_wbuf_vfunc = {
    "w_dx": lambda vd, pic, **kw: 8 * vd[pic],
    "d_dx": lambda vd, tic, **kw: 9 * vd[tic],
    "p_dx": lambda vd, **kw: 1,
}
conv_sbuf_vfunc = {
    "w_dx": lambda vd, **kw: 32,
    "d_dx": lambda vd, tic, **kw: 2 * vd[tic],
    "p_dx": lambda vd, pic, **kw: vd[pic] / 2,
}
conv_3x3_lbuf_vfunc = {
    "w_dx": lambda vd, pic, **kw: 8 * vd[pic],
    "d_dx": lambda vd, tic, iw, **kw: iw * vd[tic],
    "p_dx": lambda vd, **kw: 3,
}
conv_3x3_dw_lbuf_vfunc = {
    "w_dx": lambda vd, pic, **kw: 8 * vd[pic],
    "d_dx": lambda vd, tic, iw, **kw: 3 * iw * vd[tic],
    "p_dx": lambda vd, **kw: 1,
}
linear_wbuf_vfunc = {
    "w_dx": lambda vd, pic, **kw: 8 * vd[pic],
    "d_dx": lambda vd, tic, toc, **kw: vd[tic] * vd[toc],
    "p_dx": lambda vd, poc, **kw: vd[poc],
}
linear_sbuf_vfunc = {
    "w_dx": lambda vd, pic, linear_sum_w, **kw: linear_sum_w * vd[pic],
    "d_dx": lambda vd, tic, **kw: vd[tic],
    "p_dx": lambda vd, **kw: 1,
}
buf_vfunc_dict = {
    "conv_1x1": {"wbuf": conv_1x1_wbuf_vfunc, "sbuf": conv_sbuf_vfunc},
    "conv_3x3": {"sbuf": conv_sbuf_vfunc, "lbuf": conv_3x3_lbuf_vfunc},
    "conv_3x3_dw": {
        "wbuf": conv_3x3_dw_wbuf_vfunc,
        "sbuf": conv_sbuf_vfunc,
        "lbuf": conv_3x3_dw_lbuf_vfunc,
    },
    "linear": {"wbuf": linear_wbuf_vfunc, "sbuf": linear_sbuf_vfunc},
}


class EventNetFormulation(BaseFormulation):
    """Formulation by single layer"""

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
        self.config = config
        config.setdefault("form-obj", "lat_max")  # lat_max, lat_max+uti
        assert config["form-obj"] in ["lat_max", "lat_max+uti"]
        config.setdefault("form-even_p", True)
        config.setdefault("form-fix_top_pc", False)
        config.setdefault("form-fix_layer_pc", None)
        super().__init__(name, path, config)

    def build_var_list(self) -> None:
        var = DSEVar("obj", 1, is_obj=True, vtype="CONTINUOUS")
        self.var_list.append(var)
        var = DSEVar("lat_max", 1)
        self.var_list.append(var)
        var = DSEVar("total_dsp", 1)
        self.var_list.append(var)
        var = DSEVar("total_bram", 1)
        self.var_list.append(var)
        var = DSEVar(f"{top_lname}.tc", 1)
        self.var_list.append(var)
        var = DSEVar(f"{top_lname}.pc", 1, is_strict=True)
        self.var_list.append(var)
        for layer in self.model["layers"]:
            if layer["type"] == "conv" and layer["name"] == "conv1":
                self.build_var_list_layer(layer["name"], "conv_3x3")
            elif layer["type"] == "conv" and layer["name"] == "conv8":
                self.build_var_list_layer(layer["name"], "conv_1x1")
            elif layer["type"] == "block":
                self.build_var_list_layer(layer["name"] + "_c0", "conv_1x1")
                self.build_var_list_layer(layer["name"] + "_c1", "conv_3x3_dw")
                self.build_var_list_layer(layer["name"] + "_c2", "conv_1x1")
            elif layer["type"] == "linear":
                self.build_var_list_layer(layer["name"], "linear")
            else:
                raise TypeError(f"Unknown layer type: {layer['type']}")

    def build_var_list_layer(self, lname: str, ltype: str) -> None:
        """Build common variables for conv layers"""
        assert ltype in ["conv_1x1", "conv_3x3", "conv_3x3_dw", "linear"]
        var = DSEVar(f"{lname}.lat", 1)
        self.var_list.append(var)
        if "_dw" not in ltype:
            var = DSEVar(f"{lname}.tc", 1)
            self.var_list.append(var)
            var = DSEVar(f"{lname}.pc", 1, 32, is_strict=(ltype != "linear"))
            self.var_list.append(var)
            if self.config["form-even_p"]:
                var = DSEVar(f"{lname}.pc_d2", 1)
                self.var_list.append(var)
        var = DSEVar(f"{lname}.dsp", 1)
        self.var_list.append(var)
        for bname in buf_vfunc_dict[ltype].keys():
            for dname in buf_dname_list:
                var = DSEVar(f"{lname}.{bname}.{dname}", 1)
                self.var_list.append(var)
        var = DSEVar(f"{lname}.bram", 1)
        self.var_list.append(var)

    def build_constr_list(self) -> None:
        self.build_constr_list_fix_pc()
        # top
        lname = top_lname
        self.build_constr_list_tcpc(lname, self.model["layers"][0]["channels"][0])
        prev_lname = lname
        # layers
        for layer in self.model["layers"]:
            if layer["type"] == "conv" and layer["name"] == "conv1":
                lname = layer["name"]
                ltype = "conv_3x3"
                lc = layer["channels"][-1]
                lsparsity = layer["sparsity"]
                self.build_constr_list_layer(
                    lname, ltype, lc, lsparsity, False, layer, prev_lname
                )
                prev_lname = lname
            elif layer["type"] == "conv" and layer["name"] == "conv8":
                lname = layer["name"]
                ltype = "conv_1x1"
                lc = layer["channels"][-1]
                lsparsity = layer["sparsity"]
                self.build_constr_list_layer(
                    lname, ltype, lc, lsparsity, False, layer, prev_lname
                )
                prev_lname = lname
            elif layer["type"] == "block":
                #
                lname = layer["name"] + "_c0"
                ltype = "conv_1x1"
                lc = layer["channels"][1]
                lsparsity = layer["sparsity"][0]
                self.build_constr_list_layer(
                    lname, ltype, lc, lsparsity, False, layer, prev_lname
                )
                prev_lname = lname
                #
                lname = layer["name"] + "_c1"
                ltype = "conv_3x3_dw"
                lc = layer["channels"][1]
                lsparsity = layer["sparsity"][1] * sum(
                    layer["kernel_sparsity"][x] * (x + 1) for x in range(9)
                )
                self.build_constr_list_layer(
                    lname, ltype, lc, lsparsity, False, layer, prev_lname
                )
                # prev_lname = lname
                #
                lname = layer["name"] + "_c2"
                ltype = "conv_1x1"
                lc = layer["channels"][2]
                lsparsity = layer["sparsity"][2]
                self.build_constr_list_layer(
                    lname, ltype, lc, lsparsity, layer["residual"], layer, prev_lname
                )
                prev_lname = lname
            elif layer["type"] == "linear":
                lname = layer["name"]
                ltype = "linear"
                lc = layer["channels"][-1]
                lsparsity = 1.0
                self.build_constr_list_layer(
                    lname, ltype, lc, lsparsity, False, layer, prev_lname
                )
                prev_lname = lname
            else:
                raise TypeError(f"Unknown layer type: {layer['type']}")
        self.build_constr_list_hw()
        self.build_constr_list_obj()

    def build_constr_list_fix_pc(self) -> None:
        # fix top_pc
        if self.config["form-fix_top_pc"]:
            constr = DSEConstr(
                f"c.fix_pc.{top_lname}.pc",
                lambda vd: vd[f"{top_lname}.pc"],
                inter_var=self.model["layers"][0]["channels"][0],
            )
            self.constr_list.append(constr)
        # fix all layer pc
        if self.config["form-fix_layer_pc"] is not None:
            p = int(self.config["form-fix_layer_pc"])
            for layer in self.model["layers"]:
                lname = layer["name"]
                if layer["type"] == "conv":
                    vname_list = [f"{lname}.pc"]
                elif layer["type"] == "block":
                    vname_list = [f"{lname}_c0.pc", f"{lname}_c2.pc"]
                elif layer["type"] == "linear":
                    vname_list = [f"{lname}.pc"]
                else:
                    raise TypeError(f"Unknown layer type: {layer['type']}")
                for vname in vname_list:
                    constr = DSEConstr(
                        f"c.fix_pc.{vname}",
                        lambda vd, vname=vname: vd[vname],
                        inter_var=p,
                    )
                    self.constr_list.append(constr)

    def build_constr_list_tcpc(self, lname: str, lc: int) -> None:
        # <lname>.tc * <lname>.pc == LC
        constr = DSEConstr(
            f"c.{lname}.tc_x_pc",
            lambda vd: vd[f"{lname}.tc"] * vd[f"{lname}.pc"],
            inter_var=lc,
        )
        self.constr_list.append(constr)

    def build_constr_list_even_p(self, lname: str) -> None:
        # <lname>.pc_d2 * 2 == <lname>.pc
        if self.config["form-even_p"]:
            constr = DSEConstr(
                f"c.{lname}.pc_d2",
                lambda vd: vd[f"{lname}.pc_d2"] * 2,
                inter_var=f"{lname}.pc",
            )
            self.constr_list.append(constr)

    def build_constr_list_layer(
        self,
        lname: str,
        ltype: str,
        lc: int,
        lsparsity: float,
        lresidual: bool,
        layer: dict,
        prev_lname: str,
    ) -> None:
        input_shape = layer["input_shape"]
        output_shape = [x // layer["stride"] for x in layer["input_shape"]]
        linear_sum_w = 8 + math.ceil(math.log2(input_shape[0] * input_shape[1]))
        assert isinstance(
            lsparsity, float
        ), f"{lname}: lsparsity={lsparsity} is not float"
        # lat_max >= <lname>.lat * SPARSITY
        if ltype != "linear":  # linear layer is appended to obj
            constr = DSEConstr(
                f"c.{lname}.lat",
                lambda vd: vd["lat_max"] >= vd[f"{lname}.lat"] * lsparsity,
            )
            self.constr_list.append(constr)
        # lat_vfunc == <lname>.lat
        lat_vfunc = lambda vd: lat_vfunc_dict[ltype](
            vd, output_shape, f"{prev_lname}.tc", f"{lname}.tc"
        )
        constr = DSEConstr(f"c.{lname}.lat", lat_vfunc, inter_var=f"{lname}.lat")
        self.constr_list.append(constr)
        # <lname>.tc, <lname>.pc
        if "_dw" not in ltype:
            self.build_constr_list_tcpc(lname, lc)
            self.build_constr_list_even_p(lname)
        # dsp_vfunc <=/== <lname>.dsp
        dsp_vfunc = lambda vd: dsp_vfunc_dict[ltype](
            vd, f"{prev_lname}.pc", f"{lname}.pc"
        ) + (res_dsp_vfunc(vd, f"{prev_lname}.pc", f"{lname}.pc") if lresidual else 0)
        constr = DSEConstr(
            f"c.{lname}.dsp.gpkit",
            lambda vd: dsp_vfunc(vd) <= vd[f"{lname}.dsp"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            f"c.{lname}.dsp.scip", dsp_vfunc, inter_var=f"{lname}.dsp", scope="scip"
        )
        self.constr_list.append(constr)
        # buf_vfunc <= <lname>.<bname>.<dname>*SCALE
        for bname in buf_vfunc_dict[ltype].keys():
            for dscale, dname in zip(buf_dscale_list, buf_dname_list):
                vname = f"{lname}.{bname}.{dname}"
                vfunc = buf_vfunc_dict[ltype][bname][dname]
                buf_vfunc = (
                    lambda vd, vname=vname, vfunc=vfunc, dscale=dscale: vfunc(
                        vd,
                        pic=f"{prev_lname}.pc",
                        poc=f"{lname}.pc",
                        tic=f"{prev_lname}.tc",
                        toc=f"{lname}.tc",
                        iw=input_shape[0],
                        linear_sum_w=linear_sum_w,
                    )
                    <= vd[vname] * dscale
                )
                constr = DSEConstr(f"c.{lname}.{bname}.{dname}", buf_vfunc)
                self.constr_list.append(constr)
        # bram_vfunc=sum(prod(<lname>.<bname>.<dname>)) <=/== <lname>.bram
        bram_vfunc = lambda vd: my_sum(
            my_prod(vd[f"{lname}.{bname}.{dname}"] for dname in buf_dname_list)
            for bname in buf_vfunc_dict[ltype].keys()
        )
        constr = DSEConstr(
            f"c.{lname}.bram.gpkit",
            lambda vd: bram_vfunc(vd) <= vd[f"{lname}.bram"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            f"c.{lname}.bram.scip",
            bram_vfunc,
            inter_var=f"{lname}.bram",
            scope="scip",
        )
        self.constr_list.append(constr)

    def build_constr_list_hw(self) -> None:
        name_list = []
        for layer in self.model["layers"]:
            if layer["type"] == "conv":
                name_list.append(layer["name"])
            elif layer["type"] == "block":
                name_list.extend(layer["name"] + f"_c{i}" for i in range(3))
            elif layer["type"] == "linear":
                name_list.append(layer["name"])
            else:
                raise TypeError(f"Unknown layer type: {layer['type']}")
        # total_dsp_vfunc=sum(<lname>.dsp) <=/== total_dsp
        total_dsp_vfunc = lambda vd: my_sum(vd[f"{lname}.dsp"] for lname in name_list)
        constr = DSEConstr(
            "c.total_dsp.gpkit",
            lambda vd: total_dsp_vfunc(vd) <= vd["total_dsp"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            "c.total_dsp.scip", total_dsp_vfunc, inter_var="total_dsp", scope="scip"
        )
        self.constr_list.append(constr)
        # total_dsp <= DSP
        constr = DSEConstr("c.total_dsp", lambda vd: vd["total_dsp"] <= self.hw["dsp"])
        self.constr_list.append(constr)
        # total_bram_vfunc=sum(<lname>.bram) <=/== total_bram
        total_bram_vfunc = lambda vd: my_sum(vd[f"{lname}.bram"] for lname in name_list)
        constr = DSEConstr(
            "c.bram.gpkit",
            lambda vd: total_bram_vfunc(vd) <= vd["total_bram"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            "c.bram.scip", total_bram_vfunc, inter_var="total_bram", scope="scip"
        )
        self.constr_list.append(constr)
        # total_bram <= BRAM
        constr = DSEConstr(
            "c.total_bram", lambda vd: vd["total_bram"] <= self.hw["bram36"] * 2
        )
        self.constr_list.append(constr)

    def build_constr_list_obj(self) -> None:
        linear_lname_list = [
            layer["name"] for layer in self.model["layers"] if layer["type"] == "linear"
        ]
        if len(linear_lname_list) == 0:
            linear_lat_func = lambda vd: 0
        else:
            linear_lat_func = lambda vd: my_sum(
                vd[f"{lname}.lat"] for lname in linear_lname_list
            )
        total_lat_func = lambda vd: vd["lat_max"] + linear_lat_func(vd)
        if self.config["form-obj"] == "lat_max":
            gpkit_obj_func = total_lat_func
            scip_obj_func = total_lat_func
        elif self.config["form-obj"] == "lat_max+uti":
            gpkit_obj_func = total_lat_func
            scip_obj_func = (
                lambda vd: total_lat_func(vd)
                + vd["total_dsp"] / (self.hw["dsp"]) * 0.5
                + vd["total_bram"] / (self.hw["bram36"] * 2) * 0.5
            )
        else:
            raise ValueError(f"Unknown obj: {self.config['form-obj']}")
        constr = DSEConstr(
            "c.obj.gpkit", lambda vd: vd["obj"] >= gpkit_obj_func(vd), scope="gpkit"
        )
        self.constr_list.append(constr)
        constr = DSEConstr("c.obj.scip", scip_obj_func, inter_var="obj", scope="scip")
        self.constr_list.append(constr)

    def build_result(self) -> None:
        if self.solution is not None:
            self.result = {k: v for k, v in self.model.items() if k != "layers"}
            self.result.update(
                {
                    k: self.solution[k]
                    for k in ["obj", "lat_max", "total_dsp", "total_bram"]
                }
            )
            self.result["layers"] = deepcopy(self.model["layers"])
            prev_lname = top_lname
            for layer in self.result["layers"]:
                lname = layer["name"]
                if layer["type"] == "conv":
                    lname_list = [prev_lname, lname]
                    sub_lname_list = [f"{lname}"]
                elif layer["type"] == "block":
                    lname_list = [prev_lname, lname + "_c0", lname + "_c2"]
                    sub_lname_list = [f"{lname}_c{i}" for i in range(3)]
                elif layer["type"] == "linear":
                    lname_list = [prev_lname, lname]
                    sub_lname_list = [f"{lname}"]
                else:
                    raise TypeError(f"Unknown layer type: {layer['type']}")
                layer["parallelism"] = [self.solution[f"{l}.pc"] for l in lname_list]
                layer["lat"] = [
                    self.solution[f"{lname}.lat"] for lname in sub_lname_list
                ]
                layer["dsp"] = my_sum(
                    self.solution[f"{lname}.dsp"] for lname in sub_lname_list
                )
                layer["bram"] = my_sum(
                    self.solution[f"{lname}.bram"] for lname in sub_lname_list
                )
                prev_lname = lname_list[-1]
