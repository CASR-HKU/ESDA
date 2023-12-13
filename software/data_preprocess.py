import os
import argparse
import yaml
from dataset.loader import Loader
import utils.utils as utils

from config.settings import Settings


def main():
    parser = argparse.ArgumentParser(description='Preprocess & Visualization')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--mode', default='all', help='which set to process')
    parser.add_argument('--data_path', help='Path to dataset', default="")

    # Preprocess options
    parser.add_argument('--preprocess', action='store_true', help='Whether conduct preprocess')
    parser.add_argument('--save_dir', "-s", type=str, default='')
    parser.add_argument('--vis_hist', action='store_true', help='Visualize histogram')
    parser.add_argument('--window_size', type=float, default=1024)
    parser.add_argument('--overlap_ratio', type=float, default=0.5)
    parser.add_argument('--denoise_time', type=float, default=-1)
    parser.add_argument('--pre_type', type=str, default="time")
    parser.add_argument('--use_denoise', action='store_true', help='Whether using denoise events in preprocessed data')

    args = parser.parse_args()

    settings_filepath = args.settings_file

    with open(settings_filepath, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)
        cfg['transform']["sample"]["window"] = args.window_size
        cfg["transform"]["denoise"]["filter_time"] = args.denoise_time
    Dataset = utils.select_dataset(cfg["dataset"]["name"])

    if args.data_path != "":
        cfg["dataset"]["dataset_path"] = args.data_path

    if "ASL" in cfg["dataset"]["name"]:
        modes = ["training"]
        target_files = [os.path.join(args.save_dir, 'data.h5')]
    else:
        if args.mode == 'all':
            modes = ['training', 'validation']
            target_files = [os.path.join(args.save_dir, 'train.h5'), os.path.join(args.save_dir, 'val.h5')]
        elif args.mode == 'train':
            modes = ['training']
            target_files = [os.path.join(args.save_dir, 'train.h5')]
        elif args.mode == 'val':
            modes = ['validation']
            target_files = [os.path.join(args.save_dir, 'val.h5')]
        else:
            raise NotImplementedError

    for mode, file in zip(modes, target_files):
        file = "" if not args.save_dir else file
        print("Processing {} set".format(mode))
        dataset = Dataset(cfg, mode=mode, shuffle=False, use_denoise=args.use_denoise, cross_valid=-1)
        dataset.preprocess(args.window_size, args.overlap_ratio, save_folder=file, vis_hist=args.vis_hist,
                           preprocess_type=args.pre_type, denoise_time=args.denoise_time)
    # sys.exit(0)

    # loader = Loader(dataset, batch_size=settings.batch_size, device="cpu",
    #                 num_workers=settings.num_cpu_workers, pin_memory=False)

    # if "DVSGesture" in dataset.dataset_name:
    #     model_input_size = torch.tensor((159, 159))
    # else:
    #     model_input_size = torch.tensor((191, 255))
    #
    # for i_batch, sample_batched in enumerate(loader):
    #     _, labels, histogram = sample_batched
    #
    #     image = utils.histogram_visualize(histogram, model_input_size,
    #                                       [loader.loader.dataset.object_classes[idx] for idx in labels])
    #     cv2.imshow("show", image)
    #     cv2.waitKey(0)


if __name__ == '__main__':
    main()
