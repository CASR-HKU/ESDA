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
            self.remove_BN = False if "remove_BN" not in model else model["remove_BN"]
            self.remove_depth = 0 if "remove_depth" not in model else model['remove_depth']
            self.model_type = model['model_type'] if "model_type" in model else "base"
            self.relu_type = model['relu_type'] if "relu_type" in model else "relu6"
            self.drop_config = model['drop_config'] if "drop_config" in model else {}
            try:
                self.event_representation = settings["transform"]["toImage"]["type"]
            except:
                self.event_representation = settings['dataset']['event_representation']

            # --- dataset ---
            # dataset = settings['dataset']
            # self.dataset_name = dataset['name']
            # self.event_representation = dataset['event_representation']
            # if self.dataset_name == 'NCaltech101':
            #     dataset_specs = dataset['ncaltech101']
            # elif self.dataset_name == 'NCaltech101_ObjectDetection':
            #     dataset_specs = dataset['ncaltech101_objectdetection']
            # elif self.dataset_name == 'Prophesee':
            #     dataset_specs = dataset['prophesee']
            # elif self.dataset_name == 'NCars':
            #     dataset_specs = dataset['ncars']
            # elif self.dataset_name == 'DVSGESTRUE':
            #     dataset_specs = dataset['DVSGESTRUE']

            # self.dataset_path = dataset_specs['dataset_path']
            # assert os.path.isdir(self.dataset_path)
            # self.object_classes = dataset_specs['object_classes']
            # self.height = dataset_specs['height']
            # self.width = dataset_specs['width']
            # self.nr_events_window = dataset_specs['nr_events_window']

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']
            self.use_pretrained = checkpoint['use_pretrained']
            self.pretrained_dense_vgg = checkpoint['pretrained_dense_vgg']
            self.pretrained_sparse_vgg = checkpoint['pretrained_sparse_vgg']

            # --- optimization ---
            optimization = settings['optim']
            self.optimizer = optimization['optimizer']
            self.batch_size = optimization['batch_size']
            self.init_lr = float(optimization['init_lr'])
            self.steps_lr = optimization['steps_lr'] if "steps_lr" in optimization else optimization['decay_times']
            self.factor_lr = float(optimization['factor_lr'])
