import os
import shutil

src_path = '/vol/datastore/baoheng/eventNetHW/EDSA_correctShift/DVS_0p25_shift16-zcu102_80res/full'
dest_path = '../hw/tmp'

cfg_path = os.path.join(src_path, 'cfg.json')
cfg_dest_path = os.path.join(dest_path, 'cfg.json')
shutil.copyfile(cfg_path, cfg_dest_path)

