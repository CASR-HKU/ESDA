from typing import List, Optional
from formulation.base_formulation import BaseFormulation
from utils.arch_param import ArchParam
from utils.common import my_prod, my_sum
from utils.dse_constr import DSEConstr
from utils.dse_var import DSEVar
from utils.model_spec import ModelSpec
from utils.node_param import NodeParam
from utils.platform_spec import PlatformSpec
from utils.agna import AGNAConfig, AGNANodeEvaluation, agna_dims, agna_levels


class ArchSearch(BaseFormulation):

    platform: PlatformSpec
    model_list: List[ModelSpec]
    config: AGNAConfig
    result: Optional[ArchParam]

    def __init__(
        self,
        name: str,
        path: str,
        platform: PlatformSpec,
        model_list: List[ModelSpec],
        config: AGNAConfig,
    ) -> None:
        self.platform = platform
        self.model_list = model_list
        self.config = config
        super().__init__(name, path)

    def build_var_list(self) -> None:
        for dim in agna_dims:
            var = DSEVar(f"A_{dim}", 1, is_strict=True)
            self.var_list.append(var)
        var = DSEVar("pe_num", 1)
        self.var_list.append(var)
        var = DSEVar("dsp_pack_A_45", 1)
        self.var_list.append(var)
        var = DSEVar("dsp_num", 1)
        self.var_list.append(var)
        var = DSEVar("bram_num", 1)
        self.var_list.append(var)
        var = DSEVar("dsp_util", ub=1, vtype="CONTINUOUS")
        self.var_list.append(var)
        var = DSEVar("bram_util", ub=1, vtype="CONTINUOUS")
        self.var_list.append(var)
        for model in self.model_list:
            var = DSEVar(f"{model.name}_t", 1)
            self.var_list.append(var)
            for node in model.unique_node_iter():
                prefix = f"{model.name}_{node.name}"
                var = DSEVar(f"{prefix}_t", 1)
                self.var_list.append(var)
                var = DSEVar(f"{prefix}_c", 1)
                self.var_list.append(var)
                var = DSEVar(f"{prefix}_m", 1)
                self.var_list.append(var)
                for dim in agna_dims:
                    var = DSEVar(f"{prefix}_c_{dim}", 1)
                    self.var_list.append(var)
                for dim in agna_dims:
                    var = DSEVar(f"{prefix}_s_{dim}", 1)
                    self.var_list.append(var)
        var = DSEVar("obj", 1, is_obj=True, vtype="CONTINUOUS")
        self.var_list.append(var)

    def build_constr_list(self) -> None:
        self.build_constr_arch()
        self.build_constr_hw()
        for model in self.model_list:
            for node in model.unique_node_iter():
                prefix = f"{model.name}_{node.name}"
                self.build_constr_node(prefix, node)
        self.build_constr_obj()

    def build_constr_arch(self) -> None:
        constr = DSEConstr("opt_arch_2le3", lambda vd: vd["A_2"] <= vd["A_3"])
        self.constr_list.append(constr)
        constr = DSEConstr("opt_arch_4le5", lambda vd: vd["A_4"] <= vd["A_5"])
        self.constr_list.append(constr)
        constr = DSEConstr("opt_arch_45", lambda vd: vd["A_4"] * vd["A_5"] <= 32)
        self.constr_list.append(constr)
        constr = DSEConstr("opt_arch_pe_min", lambda vd: vd["pe_num"] >= 16)
        self.constr_list.append(constr)
        constr = DSEConstr("opt_arch_pe_max", lambda vd: vd["pe_num"] <= 128)
        self.constr_list.append(constr)

    def build_constr_hw(self) -> None:
        # packed A_4, A_5
        if self.config.dsp_pack:
            constr = DSEConstr(
                "dsp_pack_A_45",
                lambda vd: vd["A_4"] * vd["A_5"] / 2 <= vd["dsp_pack_A_45"],
            )
        else:
            constr = DSEConstr(
                "dsp_pack_A_45",
                lambda vd: vd["A_4"] * vd["A_5"],
                inter_var="dsp_pack_A_45",
            )
        self.constr_list.append(constr)
        # dsp_num
        constr = DSEConstr(
            "dsp_num",
            lambda vd: vd["A_0"]
            * vd["A_1"]
            * vd["A_2"]
            * vd["A_3"]
            * vd["dsp_pack_A_45"]
            * vd["pe_num"],
            inter_var="dsp_num",
        )
        self.constr_list.append(constr)
        # bram_num
        constr = DSEConstr(
            "bram_num",
            lambda vd: vd["bram_num"]
            >= (
                self.platform.beta[0] * vd["A_1"]
                + self.platform.beta[1] * vd["A_0"] * vd["A_1"] * vd["A_2"] * vd["A_3"]
                + self.platform.beta[2] * vd["A_0"] * vd["A_4"] * vd["A_5"]
            )
            * vd["pe_num"]
            + self.config.bram_ovhd,
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            "bram_num",
            lambda vd: (
                self.platform.beta[0] * vd["A_1"]
                + self.platform.beta[1] * vd["A_0"] * vd["A_1"] * vd["A_2"] * vd["A_3"]
                + self.platform.beta[2] * vd["A_0"] * vd["A_4"] * vd["A_5"]
            )
            * vd["pe_num"]
            + self.config.bram_ovhd,
            "bram_num",
            scope="scip",
        )
        self.constr_list.append(constr)
        # dsp_util
        constr = DSEConstr(
            "dsp_util", lambda vd: vd["dsp_num"] / self.platform.max_dsp, "dsp_util"
        )
        self.constr_list.append(constr)
        # bram_util
        constr = DSEConstr(
            "bram_util", lambda vd: vd["bram_num"] / self.platform.max_bram, "bram_util"
        )
        self.constr_list.append(constr)

    def build_constr_node(self, prefix: str, node: NodeParam) -> None:
        tk = f"{prefix}_t"
        ck = f"{prefix}_c"
        mk = f"{prefix}_m"
        sk = f"{prefix}_s"
        for dim in agna_dims:
            bnd = node.loop_bounds[dim]
            cdk = f"{ck}_{dim}"
            sdk = f"{sk}_{dim}"
            adk = f"A_{dim}"
            if node.is_dpws and dim in [0, 1]:
                constr = DSEConstr(
                    f"{prefix}_{dim}_lb",
                    lambda vd, bnd=bnd, cdk=cdk, sdk=sdk: vd[cdk] * vd[sdk] >= bnd,
                )
                self.constr_list.append(constr)
                constr = DSEConstr(
                    f"{prefix}_{dim}_ub",
                    lambda vd, bnd=bnd, cdk=cdk, sdk=sdk: vd[cdk] * vd[sdk]
                    <= bnd + vd[sdk] - 1,
                    scope="scip",
                )
                self.constr_list.append(constr)
            else:
                constr = DSEConstr(
                    f"{prefix}_{dim}_lb",
                    lambda vd, bnd=bnd, cdk=cdk, sdk=sdk, adk=adk: vd[cdk]
                    * vd[sdk]
                    * vd[adk]
                    >= bnd,
                )
                self.constr_list.append(constr)
                constr = DSEConstr(
                    f"{prefix}_{dim}_ub",
                    lambda vd, bnd=bnd, cdk=cdk, sdk=sdk, adk=adk: vd[cdk]
                    * vd[sdk]
                    * vd[adk]
                    <= bnd + vd[sdk] * vd[adk] - 1,
                    scope="scip",
                )
                self.constr_list.append(constr)
            if self.config.skip_s:
                # sdk=1
                constr = DSEConstr(f"{prefix}_{dim}_s", lambda vd: 1, sdk)
                self.constr_list.append(constr)
        if self.config.skip_s:
            # c = prod(cd)/pe_num
            constr = DSEConstr(
                ck,
                lambda vd, ck=ck: my_prod(vd[f"{ck}_{dim}"] for dim in agna_dims)
                / vd["pe_num"],
                ck,
            )
            self.constr_list.append(constr)
        else:
            # c = prod(cd)
            constr = DSEConstr(
                ck,
                lambda vd, ck=ck: my_prod(vd[f"{ck}_{dim}"] for dim in agna_dims),
                ck,
            )
            self.constr_list.append(constr)
            # pe_num >= prod(sd)
            constr = DSEConstr(
                f"{prefix}_pe_num",
                lambda vd, sk=sk: vd["pe_num"]
                >= my_prod(vd[f"{sk}_{dim}"] for dim in agna_dims),
            )
            self.constr_list.append(constr)
        # m
        node_eval = AGNANodeEvaluation(self.platform, node)
        c_comm = node_eval.get_c_comm()
        constr = DSEConstr(mk, lambda vd: c_comm, mk)
        self.constr_list.append(constr)
        # t
        constr = DSEConstr(f"{tk}_c", lambda vd, tk=tk, ck=ck: vd[tk] >= vd[ck])
        self.constr_list.append(constr)
        constr = DSEConstr(f"{tk}_m", lambda vd, tk=tk, mk=mk: vd[tk] >= vd[mk])
        self.constr_list.append(constr)

    def build_constr_obj(self) -> None:
        for model in self.model_list:
            # model_t >= sum(cnt*node_t)
            mn = model.name
            constr = DSEConstr(
                f"{mn}_t",
                lambda vd, mn=mn: vd[f"{mn}_t"]
                >= my_sum(
                    node.unique_cnt * vd[f"{mn}_{node.name}_t"]
                    for node in model.unique_node_iter()
                ),
                scope="gpkit",
            )
            self.constr_list.append(constr)
            constr = DSEConstr(
                f"{mn}_t",
                lambda vd, mn=mn: my_sum(
                    node.unique_cnt * vd[f"{mn}_{node.name}_t"]
                    for node in model.unique_node_iter()
                ),
                f"{mn}_t",
                scope="scip",
            )
            self.constr_list.append(constr)
        # obj >= sum(model_t)
        constr = DSEConstr(
            "obj",
            lambda vd: vd["obj"]
            >= my_sum(vd[f"{model.name}_t"] for model in self.model_list),
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            "obj",
            lambda vd: my_sum(vd[f"{model.name}_t"] for model in self.model_list),
            "obj",
            scope="scip",
        )
        self.constr_list.append(constr)

    def build_result(self) -> None:
        if self.solution is not None:
            arch_list = [self.solution[f"A_{dim}"] for dim in agna_dims]
            self.result = ArchParam(A=arch_list, pe_num=self.solution["pe_num"])
