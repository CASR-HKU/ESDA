import os
import time
import yaml
import torch
import shutil


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings['hardware']
            gpu_device = hardware['gpu_device']

            self.gpu_device = torch.device("cpu") if gpu_device == "cpu" else torch.device("cuda:" + str(gpu_device))

            self.num_cpu_workers = hardware['num_cpu_workers']
            if self.num_cpu_workers < 0:
                self.num_cpu_workers = os.cpu_count()

            # --- Model ---
            model = settings['model']
            self.model_name = model['model_name']
            self.width_mult = model['width_mult'] if "mobilenet" in self.model_name else None
            self.remove_depth = 0 if "remove_depth" not in model else model['remove_depth']
            self.model_type = model['model_type'] if "model_type" in model else "base"
            self.relu_type = model['relu_type'] if "relu_type" in model else "relu6"
            self.event_representation = settings["transform"]["toImage"]["type"]

            # --- optimization ---
            optimization = settings['optim']
            self.optimizer = optimization['optimizer']
            self.batch_size = optimization['batch_size']
            self.init_lr = float(optimization['init_lr'])
            self.steps_lr = optimization['steps_lr'] if "steps_lr" in optimization else optimization['decay_times']
            self.factor_lr = float(optimization['factor_lr'])
