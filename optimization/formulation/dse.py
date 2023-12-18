import csv
import os
from typing import List
from formulation.arch_search import ArchSearch
from formulation.op_schedule import OpSchedule
from formulation.mlpmixer_schedule import MlpMixerSchedule
from formulation.simp_formulation import SimpFormulation
from utils.agna import AGNAConfig
from utils.platform_spec import PlatformSpec
from utils.model_spec import ModelSpec


class DSE:
    platform: PlatformSpec
    model_list: List[ModelSpec]
    path: str

    def __init__(
        self, platform: PlatformSpec, model_list: List[ModelSpec], path: str
    ) -> None:
        self.platform = platform
        self.model_list = model_list
        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def run(self) -> None:
        self.run_arch_search()

    def run_arch_search(self) -> None:
        dflt_config = AGNAConfig()
        dflt_config.save(os.path.join(self.path, "as-config.json"))
        as_i = ArchSearch("as", self.path, self.platform, self.model_list, dflt_config)
        as_i.solve()
        if as_i.result is not None:
            as_i.result.save(os.path.join(self.path, "as-result.json"))

    def run_op_schedule(self) -> None:
        for model in self.model_list:
            for node in model.unique_node_iter():
                os_i = OpSchedule("os", self.path)
                os_i.solve()

    def run_simp(self) -> None:
        sf = SimpFormulation("sf", self.path)
        sf.solve()
        if sf.result is not None:
            sf.result.save(os.path.join(self.path, "sf-result.json"))
