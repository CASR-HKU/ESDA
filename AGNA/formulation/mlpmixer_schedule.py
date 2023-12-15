from typing import Any, Dict, List
from formulation.base_formulation import BaseFormulation
from utils.dse_constr import DSEConstr
from utils.dse_var import DSEVar
from utils.model_spec import ModelSpec
from utils.platform_spec import PlatformSpec


class MlpMixerSchedule(BaseFormulation):

    platform: PlatformSpec
    fixed_sa_size: int
    model: Dict[str, Any]

    def __init__(
        self,
        name: str,
        path: str,
        platform: PlatformSpec,
        model: Dict[str, Any],
        fixed_sa_size=-1,
    ) -> None:
        self.platform = platform
        self.model = model
        self.fixed_sa_size = fixed_sa_size
        super().__init__(name, path)

    def build_var_list(self) -> None:
        # sa shape
        var = DSEVar("SA_SIZE", 1, is_strict=True)
        self.var_list.append(var)
        for dname in ["C", "S", "D_S", "D_C"]:
            for level in ["P", "Q", "R"]:
                var = DSEVar(f"schdl_{level}_{dname}", 1, is_strict=True)
                self.var_list.append(var)
        for data in ["f1", "w1", "f2", "w2", "f3"]:
            for blk in ["tm", "cm"]:
                var = DSEVar(f"{data}_tile_size_{blk}", 1)
                self.var_list.append(var)
        # buffer size
        for data in ["f1", "w1", "f2", "w2", "f3"]:
            var = DSEVar(f"{data}_buf_size", 1)
            self.var_list.append(var)
        # hw utilization
        var = DSEVar("dsp_num", 1)
        self.var_list.append(var)
        var = DSEVar("bram_num", 1)
        self.var_list.append(var)
        var = DSEVar("dsp_util", ub=1, vtype="CONTINUOUS")
        self.var_list.append(var)
        var = DSEVar("bram_util", ub=1, vtype="CONTINUOUS")
        self.var_list.append(var)
        # latency
        for blk in ["tm", "cm"]:
            var = DSEVar(f"lat_{blk}_f", 1)  # feature
            self.var_list.append(var)
            var = DSEVar(f"lat_{blk}_w", 1)  # weight
            self.var_list.append(var)
            var = DSEVar(f"lat_{blk}_c", 1)  # compute
            self.var_list.append(var)
            var = DSEVar(f"lat_{blk}_beg", 1)
            self.var_list.append(var)
            var = DSEVar(f"lat_{blk}_mid", 1)
            self.var_list.append(var)
            var = DSEVar(f"lat_{blk}_end", 1)
            self.var_list.append(var)
            var = DSEVar(f"lat_{blk}", 1)
            self.var_list.append(var)
        var = DSEVar("lat", 1)
        self.var_list.append(var)
        var = DSEVar("obj", 1, is_obj=True, vtype="CONTINUOUS")
        self.var_list.append(var)

    def build_constr_list(self) -> None:
        self.build_constr_schd_factor()
        for blk, dname_list in zip(
            ["tm", "cm"], [["C", "S", "D_S"], ["S", "C", "D_C"]]
        ):
            self.build_constr_tile_size(blk, dname_list)
            self.build_constr_buf_size(blk, dname_list)
            self.build_constr_lat(blk, dname_list)
        self.build_constr_hw_util()
        self.build_constr_obj()

    def build_constr_schd_factor(self) -> None:
        """Schedule factor constraints (common)."""
        #       |   S  |   C  |  D_S |  D_C
        # ------+------+------+------+------
        #  P    | P_S  | P_C  | P_D_S| P_D_C
        #  Q    | Q_S  | Q_C  | Q_D_S| Q_D_C
        #  R    | R_S  | R_C  | R_D_S| R_D_C
        #  R->SA|  SA  |  SA  |  SA  |  SA
        for dname in ["S", "C", "D_S", "D_C"]:
            # loop bound
            constr = DSEConstr(
                f"loop_bound_{dname}",
                lambda vd, dname=dname: vd[f"schdl_P_{dname}"]
                * vd[f"schdl_Q_{dname}"]
                * vd[f"schdl_R_{dname}"]
                >= self.model[dname],
            )
            self.constr_list.append(constr)
            # SA
            constr = DSEConstr(
                f"sa_{dname}",
                lambda vd, dname=dname: vd[f"schdl_R_{dname}"],
                inter_var="SA_SIZE",
            )
            self.constr_list.append(constr)
        # fixed SA
        if self.fixed_sa_size > 1:
            constr = DSEConstr(
                "sa_fixed",
                lambda vd: vd["SA_SIZE"] == self.fixed_sa_size,
            )
            self.constr_list.append(constr)

    def build_constr_tile_size(self, blk, dname_list) -> None:
        """Schedule factor constraints (common)."""
        #  fc0  |  D0  |  D1  |  D2
        # ------+------+------+------
        #  P    |  P0  |  P1  |  P2
        #  Q    |  Q0  |   1  |  Q2
        #  R    |  R0  |Q1*R1 |  R2
        # ---------------------------
        #  fc1  |  D0  |  D2  |  D1
        # ------+------+------+------
        #  P    |  P0  |  P2  |  P1
        #  Q    |  Q0  |   1  |  Q1
        #  R    |  R0  |Q2*R2 |  R1
        # ---------------------------
        mkn_list0 = dname_list
        mkn_list1 = [dname_list[i] for i in [0, 2, 1]]
        for fc, mkn_list in zip([0, 1], [mkn_list0, mkn_list1]):
            pass
        # data |  shape
        #  f1  | D0 * D1
        #  w1  | D1 * D2
        #  f2  | D0 * D2
        #  w2  | D2 * D1
        #  f3  | D0 * D1
        for data, d_idx in zip(
            ["f1", "w1", "f2", "w2", "f3"], [[0, 1], [1, 2], [0, 2], [2, 1], [0, 1]]
        ):
            dname0 = dname_list[d_idx[0]]
            dname1 = dname_list[d_idx[1]]
            constr = DSEConstr(
                f"{data}_tile_size_{blk}",
                lambda vd, blk=blk, dname0=dname0, dname1=dname1: vd[
                    f"schdl_Q_{dname0}"
                ]
                * vd[f"schdl_R_{dname0}"]
                * vd[f"schdl_Q_{dname1}"]
                * vd[f"schdl_R_{dname1}"],
                inter_var=f"{data}_tile_size_{blk}",
            )
            self.constr_list.append(constr)

    def build_constr_buf_size(self, blk, dname_list) -> None:
        """Buffer size constraints (dataflow specific)."""
        dname0, dname1, dname2 = dname_list
        # buf_f1, f1_tile * 1 * P_D1
        constr = DSEConstr(
            f"f1_buf_size_{blk}",
            lambda vd, blk=blk, dname=dname1: vd["f1_buf_size"]
            >= vd[f"f1_tile_size_{blk}"] * vd[f"schdl_P_{dname}"],
        )
        self.constr_list.append(constr)
        # buf_w1, w1_tile * P_D1 * 1
        constr = DSEConstr(
            f"w1_buf_size_{blk}",
            lambda vd, blk=blk, dname=dname1: vd["w1_buf_size"]
            >= vd[f"w1_tile_size_{blk}"] * vd[f"schdl_P_{dname}"],
        )
        self.constr_list.append(constr)
        # buf_f2, f2_tile * 1 * 1
        constr = DSEConstr(
            f"f2_buf_size_{blk}",
            lambda vd, blk=blk: vd["f2_buf_size"] >= vd[f"f2_tile_size_{blk}"],
        )
        self.constr_list.append(constr)
        # buf_w2, w2_tile * 1 * P_D1
        constr = DSEConstr(
            f"w2_buf_size_{blk}",
            lambda vd, blk=blk, dname=dname1: vd["w2_buf_size"]
            >= vd[f"w2_tile_size_{blk}"] * vd[f"schdl_P_{dname}"],
        )
        self.constr_list.append(constr)
        # buf_f3, f3_tile * 1 * P_D1
        constr = DSEConstr(
            f"f3_buf_size_{blk}",
            lambda vd, blk=blk, dname=dname1: vd["f3_buf_size"]
            >= vd[f"f3_tile_size_{blk}"] * vd[f"schdl_P_{dname}"],
        )
        self.constr_list.append(constr)

    def build_constr_lat(self, blk, dname_list) -> None:
        """Latency constraints (dataflow-specific)."""
        dname0, dname1, dname2 = dname_list
        # feature lat, f_tile * 1 * P_D1 * data_width / dbus_width
        # f1 should be the same as f3
        constr = DSEConstr(
            f"lat_{blk}_f",
            lambda vd, blk=blk, dname=dname1: vd[f"f1_tile_size_{blk}"]
            * vd[f"schdl_P_{dname}"]
            * self.platform.data_width
            / self.platform.dbus_width,
            inter_var=f"lat_{blk}_f",
        )
        self.constr_list.append(constr)
        # weight lat, w_tile * P_D1 * 1 * data_width / dbus_width
        constr = DSEConstr(
            f"lat_{blk}_w",
            lambda vd, blk=blk, dname=dname1: vd[f"w1_tile_size_{blk}"]
            * vd[f"schdl_P_{dname}"]
            * self.platform.data_width
            / self.platform.dbus_width,
            f"lat_{blk}_w",
        )
        self.constr_list.append(constr)
        # compute lat, Q_D0 * Q_D1 * R_D1 * Q_D2
        # R_D0 and R_D2 are parallelized
        constr = DSEConstr(
            f"lat_{blk}_c",
            lambda vd, dname_list=dname_list: vd[f"schdl_Q_{dname_list[0]}"]
            * vd[f"schdl_Q_{dname_list[1]}"]
            * vd[f"schdl_R_{dname_list[1]}"]  # modified
            * vd[f"schdl_Q_{dname_list[2]}"],
            inter_var=f"lat_{blk}_c",
        )
        self.constr_list.append(constr)
        # beg lat, max(f, w)
        constr = DSEConstr(
            f"lat_{blk}_beg",
            lambda vd, blk=blk: vd[f"lat_{blk}_beg"] >= vd[f"lat_{blk}_f"],
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            f"lat_{blk}_beg",
            lambda vd, blk=blk: vd[f"lat_{blk}_beg"] >= vd[f"lat_{blk}_w"],
        )
        self.constr_list.append(constr)
        # mid lat, max(f, w*P_D2, c*P_D2*P_D1)
        constr = DSEConstr(
            f"lat_{blk}_mid",
            lambda vd, blk=blk: vd[f"lat_{blk}_mid"] >= vd[f"lat_{blk}_f"],
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            f"lat_{blk}_mid",
            lambda vd, blk=blk, dname2=dname2: vd[f"lat_{blk}_mid"]
            >= vd[f"lat_{blk}_w"] * vd[f"schdl_P_{dname2}"],
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            f"lat_{blk}_mid",
            lambda vd, blk=blk, dname1=dname1, dname2=dname2: vd[f"lat_{blk}_mid"]
            >= vd[f"lat_{blk}_c"] * vd[f"schdl_P_{dname1}"] * vd[f"schdl_P_{dname2}"],
        )
        self.constr_list.append(constr)
        # end lat, c*P_D1
        constr = DSEConstr(
            f"lat_{blk}_end",
            lambda vd, blk=blk, dname1=dname1: vd[f"lat_{blk}_c"]
            * vd[f"schdl_P_{dname1}"],
            inter_var=f"lat_{blk}_end",
        )
        self.constr_list.append(constr)
        # total lat, beg + mid*P_D0 + end
        constr = DSEConstr(
            f"lat_{blk}_gpkit",
            lambda vd, blk=blk: vd[f"lat_{blk}"]
            >= vd[f"lat_{blk}_beg"]
            + vd[f"lat_{blk}_mid"] * vd[f"schdl_P_{dname0}"]
            + vd[f"lat_{blk}_end"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            f"lat_{blk}_scip",
            lambda vd, blk=blk: vd[f"lat_{blk}_beg"]
            + vd[f"lat_{blk}_mid"] * vd[f"schdl_P_{dname0}"]
            + vd[f"lat_{blk}_end"],
            inter_var=f"lat_{blk}",
            scope="scip",
        )
        self.constr_list.append(constr)

    def build_constr_hw_util(self) -> None:
        """Hardware utilization constraints (common)."""
        # dsp number
        constr = DSEConstr(
            "dsp_num",
            lambda vd: 2 * self.platform.alpha * vd["SA_SIZE"] * vd["SA_SIZE"],
            inter_var="dsp_num",
        )
        self.constr_list.append(constr)
        # dsp util
        constr = DSEConstr(
            "dsp_util",
            lambda vd: vd["dsp_num"] / self.platform.max_dsp,
            inter_var="dsp_util",
        )
        self.constr_list.append(constr)
        # bram number
        # (f1 + w1 + f2 + w2 + f3) * data_width / 16384 or 32768
        constr = DSEConstr(
            "bram_num_gpkit",
            lambda vd: vd["bram_num"]
            >= (
                vd["f1_buf_size"]
                + vd["w1_buf_size"]
                + vd["f2_buf_size"]
                + vd["w2_buf_size"]
                + vd["f3_buf_size"]
            )
            * self.platform.data_width
            / 16384,
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            "bram_num_scip",
            lambda vd: (
                vd["f1_buf_size"]
                + vd["w1_buf_size"]
                + vd["f2_buf_size"]
                + vd["w2_buf_size"]
                + vd["f3_buf_size"]
            )
            * self.platform.data_width
            / 16384,
            inter_var="bram_num",
            scope="scip",
        )
        self.constr_list.append(constr)
        # bram util
        constr = DSEConstr(
            "bram_util",
            lambda vd: vd["bram_num"] / self.platform.max_bram,
            inter_var="bram_util",
        )
        self.constr_list.append(constr)

    def build_constr_obj(self) -> None:
        """Objective function (common)."""
        # # lat, max(lat_tm, lat_cm)
        # constr = DSEConstr(
        #     "lat",
        #     lambda vd: vd["lat"] >= vd["lat_tm"],
        # )
        # self.constr_list.append(constr)
        # constr = DSEConstr(
        #     "lat",
        #     lambda vd: vd["lat"] >= vd["lat_cm"],
        # )
        # self.constr_list.append(constr)
        # lat, lat_tm + lat_cm
        constr = DSEConstr(
            "lat_gpkit",
            lambda vd: vd["lat"] >= vd["lat_tm"] + vd["lat_cm"],
            scope="gpkit",
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            "lat_scip",
            lambda vd: vd["lat_tm"] + vd["lat_cm"],
            inter_var="lat",
            scope="scip",
        )
        self.constr_list.append(constr)
        # obj
        constr = DSEConstr(
            "obj_gpkit", lambda vd: vd["lat"], inter_var="obj", scope="gpkit"
        )
        self.constr_list.append(constr)
        constr = DSEConstr(
            "obj_scip",
            lambda vd: vd["lat"] + vd["bram_util"],
            inter_var="obj",
            scope="scip",
        )
        self.constr_list.append(constr)

    def print_solution(self) -> None:
        sol = self.solution
        sol_tbl = ""
        if sol is None:
            sol_tbl += "No solution."
        else:
            cw = 8  # cell width
            sol_tbl += "\nSolution:"
            v_list = ["C", "S", "D_S", "D_C"]
            sol_tbl += f"\n{'':>{cw}}|{v_list[0]:>{cw}}{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}"
            v_list = [self.model[dname] for dname in ["C", "S", "D_S", "D_C"]]
            sol_tbl += f"\n{'':>{cw}}|{v_list[0]:>{cw}}{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}"
            sol_tbl += f"\n{'-'*cw}+{'-'*cw*4}"
            v_list = [
                sol[f"schdl_P_{dname}"]
                * sol[f"schdl_Q_{dname}"]
                * sol[f"schdl_R_{dname}"]
                for dname in ["C", "S", "D_S", "D_C"]
            ]
            sol_tbl += f"\n{'PQR':>{cw}}|{v_list[0]:>{cw}}{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}"
            for level in ["P", "Q", "R"]:
                v_list = [
                    sol[f"schdl_{level}_{dname}"] for dname in ["C", "S", "D_S", "D_C"]
                ]
                sol_tbl += f"\n{level:>{cw}}|{v_list[0]:>{cw}}{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}"
            theo_comm_lat_list = [0] * 4
            theo_comp_lat = 0
            for blk, dname_list in zip(
                ["tm", "cm"], [["C", "S", "D_S"], ["S", "C", "D_C"]]
            ):
                d0, d1, d2 = [self.model[dname] for dname in dname_list]
                p_d0, p_d1, p_d2 = [sol[f"schdl_P_{dname}"] for dname in dname_list]
                # head
                sol_tbl += "\n"
                v_list = [
                    blk,
                ] + dname_list
                sol_tbl += f"\n{v_list[0]:>{cw}}|{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}"
                sol_tbl += f"\n{'-'*cw}+{'-'*cw*3}"
                for level in ["P", "Q", "R"]:
                    v_list = [sol[f"schdl_{level}_{dname}"] for dname in dname_list]
                    sol_tbl += f"\n{level:>{cw}}|{v_list[0]:>{cw}}{v_list[1]:>{cw}}{v_list[2]:>{cw}}"
                # head
                sol_tbl += "\n"
                v_list = [blk, "rd_f1", "rd_w1", "rd_w2", "wr_f3", "comp"]
                sol_tbl += f"\n{v_list[0]:>{cw}}|{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}{v_list[4]:>{cw}}{v_list[5]:>{cw}}"
                sol_tbl += f"\n{'-'*cw}+{'-'*cw*5}"
                # theo_comm_size: f1, w1, w2, f3
                theo_comm_size_blk_list = [
                    d0 * d1,
                    d1 * d2,
                    d2 * d1,
                    d0 * d1,
                ]
                # theo_comm_lat
                theo_comm_lat_blk_list = [
                    int(s * self.platform.data_width / self.platform.dbus_width)
                    for s in theo_comm_size_blk_list
                ]
                # theo comp
                theo_comp_lat_blk = int(
                    d0 * d1 * d2 / (sol["SA_SIZE"] * sol["SA_SIZE"])
                )
                v_list = [f"theo"] + theo_comm_lat_blk_list + [theo_comp_lat_blk]
                sol_tbl += f"\n{v_list[0]:>{cw}}|{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}{v_list[4]:>{cw}}{v_list[5]:>{cw}}"
                # schdl_comm_lat: f1, w1, w2, f3
                schdl_comm_lat_blk_list = [
                    p_d0 * sol[f"lat_{blk}_f"],  # f1, P_D0*lat_f
                    p_d0 * p_d2 * sol[f"lat_{blk}_w"],  # w1, P_D0*P_D2*lat_w
                    p_d0 * p_d2 * sol[f"lat_{blk}_w"],  # w2, P_D0*P_D2*lat_w
                    p_d0 * sol[f"lat_{blk}_f"],  # f3, P_D0*lat_f
                ]
                # schdl comp: P_D0*P_D2*P_D1*lat_c
                schdl_comp_lat_blk = p_d0 * p_d2 * p_d1 * sol[f"lat_{blk}_c"]
                v_list = [f"schdl"] + schdl_comm_lat_blk_list + [schdl_comp_lat_blk]
                sol_tbl += f"\n{v_list[0]:>{cw}}|{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}{v_list[4]:>{cw}}{v_list[5]:>{cw}}"
                # schdl lat
                sol_tbl += f"\nSchdl f, w, c lat({blk}): "
                sol_tbl += f"{sol[f'lat_{blk}_f']}, "
                sol_tbl += f"{sol[f'lat_{blk}_w']}, "
                sol_tbl += f"{sol[f'lat_{blk}_c']}"
                # schdl pipeline lat
                sol_tbl += f"\nPipeline lat({blk}): "
                sol_tbl += f"{sol[f'lat_{blk}_beg']} + "
                sol_tbl += f"{sol[f'schdl_P_{dname_list[0]}']} * "
                sol_tbl += f"{sol[f'lat_{blk}_mid']} + "
                sol_tbl += f"{sol[f'lat_{blk}_end']}"
                sol_tbl += f" = {sol[f'lat_{blk}']}"
                # add to total
                theo_comm_lat_list = [
                    theo_comm_lat_list[i] + theo_comm_lat_blk_list[i] for i in range(4)
                ]
                theo_comp_lat += theo_comp_lat_blk
            theo_lat = max(theo_comm_lat_list + [theo_comp_lat])
            sol_tbl += "\n"
            v_list = ["total", "rd_f1", "rd_w1", "rd_w2", "wr_f3", "comp"]
            sol_tbl += f"\n{v_list[0]:>{cw}}|{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}{v_list[4]:>{cw}}{v_list[5]:>{cw}}"
            sol_tbl += f"\n{'-'*cw}+{'-'*cw*5}"
            v_list = [f"theo"] + theo_comm_lat_list + [theo_comp_lat]
            sol_tbl += f"\n{v_list[0]:>{cw}}|{v_list[1]:>{cw}}{v_list[2]:>{cw}}{v_list[3]:>{cw}}{v_list[4]:>{cw}}{v_list[5]:>{cw}}"
            sol_tbl += f"\n"
            sol_tbl += f"\nSchedule lat: {sol['lat']} ({sol['lat']/theo_lat:.3f}x)"
            sol_tbl += f"\nSA: {sol['SA_SIZE']}*{sol['SA_SIZE']}*{self.platform.alpha}"
            sol_tbl += f"\nDSP: {sol['dsp_num']}/{self.platform.max_dsp} = {sol['dsp_util']:.3f}"
            sol_tbl += f"\nBRAM: {sol['bram_num']}/{self.platform.max_bram} = {sol['bram_util']:.3f}"
        self.logger.info(sol_tbl)
        # print(sol_tbl)
