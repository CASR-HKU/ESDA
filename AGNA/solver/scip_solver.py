import contextlib
from functools import reduce
import os
import subprocess
from typing import Any, Dict, List
import pyscipopt
from utils.dse_constr import DSEConstr
from utils.dse_var import DSEVar
from solver.base_solver import BaseSolver


class ScipSolver(BaseSolver):
    scip_var_dict: Dict[str, pyscipopt.scip.Variable]  # type: ignore
    scip_model: pyscipopt.Model
    scip_model_path: str
    scip_log_path: str
    scip_sol_path: str

    def __init__(
        self,
        name: str,
        path: str,
        var_list: List[DSEVar],
        constr_list: List[DSEConstr],
        config: Dict[str, Any],
    ) -> None:
        config.setdefault("scip-use_cmd", False)
        config.setdefault("scip-maxnthreads", 8)
        config.setdefault("scip-timelimits", None)
        config.setdefault("scip-quiet", True)
        super().__init__(name, path, var_list, constr_list, config)

    def build(self, var_list: List[DSEVar], constr_list: List[DSEConstr]) -> None:
        self.scip_var_dict = {}
        scip_obj = None
        # init model
        self.scip_model = pyscipopt.Model()
        # add variables
        for var in var_list:
            if var.in_scip:
                self.logger.debug(f"Adding var: {var}")
                self.scip_var_dict[var.name] = self.scip_model.addVar(
                    var.name, var.vtype, var.lb, var.ub
                )
                if var.is_obj:
                    scip_obj = self.scip_var_dict[var.name]
        # add constraints
        for constr in constr_list:
            if constr.in_scip:
                self.logger.debug(f"Adding constr: {constr}")
                constr_expr = constr.get_constr_expr(self.scip_var_dict)
                self.scip_model.addCons(constr_expr, constr.name)
        # set objective
        assert scip_obj is not None, "Objective is not defined."
        self.scip_model.setObjective(scip_obj)
        # set path
        self.scip_model_path = os.path.join(self.path, f"{self.name}.cip")
        self.scip_log_path = os.path.join(self.path, f"{self.name}.log")
        self.scip_sol_path = os.path.join(self.path, f"{self.name}.sol")
        # save model
        with contextlib.redirect_stdout(None):
            self.scip_model.writeProblem(self.scip_model_path)

    def solve(self) -> None:
        use_cmd = self.config["scip-use_cmd"]
        maxnthreads = self.config["scip-maxnthreads"]
        timelimits = self.config["scip-timelimits"]
        quiet = self.config["scip-quiet"]
        open(self.scip_log_path, "w").write("")  # clear scip log
        if use_cmd:
            scip_path = os.path.join(os.environ["SCIPOPTDIR"], "bin/scip")
            assert os.path.exists(scip_path), f"scip_path: {scip_path} does not exist."
            scip_cmd = []
            scip_cmd.append(scip_path)
            scip_cmd.extend(["-l", self.scip_log_path])
            scip_cmd.extend(["-c", f"read {self.scip_model_path}"])
            if maxnthreads > 1:
                scip_cmd.extend(["-c", f"set parallel maxnthreads {maxnthreads}"])
            if timelimits is not None:
                scip_cmd.extend(["-c", f"set limits time {timelimits}"])
            scip_cmd.extend(["-c", "set write printzeros TRUE"])
            if maxnthreads > 1:
                scip_cmd.extend(["-c", "concurrentopt"])
            else:
                scip_cmd.extend(["-c", "optimize"])
            scip_cmd.extend(["-c", f"write solution {self.scip_sol_path}"])
            scip_cmd.extend(["-c", "quit"])
            scip_cmd_print = scip_cmd[0]
            scip_cmd_iter = iter(scip_cmd[1:])
            for scip_cmd_i in scip_cmd_iter:
                scip_cmd_print += f' {scip_cmd_i} "{next(scip_cmd_iter)}"'
            self.logger.info(f"Running: {scip_cmd_print}")
            try:
                timeout = None if timelimits is None else timelimits + maxnthreads + 10
                # if not os.path.exists(self.scip_sol_path):
                proc = subprocess.run(
                    scip_cmd, capture_output=quiet, timeout=timeout, check=True
                )
                first_line = open(self.scip_sol_path).readline()
                if "infeasible" in first_line:
                    raise ValueError(f"checkSol: infeasible")
                scip_sol = self.scip_model.readSolFile(self.scip_sol_path)
                # trySol() will cause segmentation fault
                if self.scip_model.checkSol(scip_sol):
                    self.logger.info(f"checkSol: valid")
                elif self.scip_model.checkSol(
                    scip_sol, printreason=False, checkintegrality=False
                ):
                    self.logger.warning(f"checkSol: integrality violated")
                else:
                    self.logger.warning(f"checkSol: other violation")
            except Exception as e:
                if isinstance(e, subprocess.TimeoutExpired):
                    self.logger.error(f"timeout after {e.timeout} seconds.")
                elif isinstance(e, subprocess.CalledProcessError):
                    self.logger.error(f"returncode: {e.returncode}")
                    self.logger.error(f"stdout: {e.stdout}")
                    self.logger.error(f"stderr: {e.stderr}")
                else:
                    self.logger.exception(e)
                self.solution = None
                self.status = "fail"
                return
        else:  # use pyscipopt, but solveConcurrent() is not working
            if quiet:
                self.scip_model.hideOutput()
            params = {}
            if maxnthreads > 1:
                params["parallel/maxnthreads"] = maxnthreads
            if timelimits is not None:
                params["limits/time"] = timelimits
            self.scip_model.setParams(params)
            self.scip_model.setLogfile(self.scip_log_path)
            self.scip_model.optimize()
            scip_sol = self.scip_model.getBestSol()
            self.scip_model.writeSol(scip_sol, self.scip_sol_path, write_zeros=True)
        # convert scip_sol to solution
        sol_dict = {}
        for var in self.scip_model.getVars():
            val = self.scip_model.getSolVal(scip_sol, var)
            if var.vtype() == "INTEGER":
                sol_dict[var.name] = int(round(val))
            else:
                sol_dict[var.name] = val
        self.solution = sol_dict
        self.status = "done"
