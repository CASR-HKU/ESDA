import math
import os
import sys

import cv2
import numpy as np
import torch.nn as nn
import tqdm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torch.utils.data.distributed
import argparse
from models.mink_mobilenetv2 import MobileNetV2ME
from models.mink_resnet import resnet18
from models.HAWQ_mobilenetv2 import Q_MobileNetV2
from models.mink_mobilenetv2_int import MobileNetV2MEInt
from models.HAWQ_quant_module.quant_modules import freeze_model, unfreeze_model
from models.mobilenet_base import mobilenet_v2
try:
    use_tb = True
    from torch.utils.tensorboard import SummaryWriter
except:
    use_tb = False
# from dataset.dataset import Dataset
import yaml
from dataset.loader import Loader
import utils.utils as utils
from utils import logger
import utils.visualizations as visualizations
from utils.bn_fold import fuse_bn_recursively
from config.settings import Settings
import MinkowskiEngine as ME
from utils.loss import Loss


# torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--save_dir', "-s", type=str, default='')
    parser.add_argument('--evaluate', '-e', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--fixBN_ratio', type=float, default=0.3, help="When to fix BN during quantization training")
    parser.add_argument('--data_path', help='Path to dataset', default="")

    parser.add_argument('--dataset_percentage', type=float, default=1)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--conv1_level', type=int, default=8)
    parser.add_argument('--shift_bit', type=int, default=31)
    parser.add_argument('--bias_bit', type=int, default=32)
    parser.add_argument('--load', type=str, default='', help='load model path')
    parser.add_argument('--auto_resume', action='store_true', help='Resume automatically')
    parser.add_argument('--gen_meta', action='store_true', help='Generate DVS slice meta')
    parser.add_argument('--min_event', type=int, default=0)
    parser.add_argument('--generate_int_model', action='store_true', help='No BN loading in the model')
    parser.add_argument('--ana_file', default=None, help='Path for error analyser')
    parser.add_argument('--lamb', type=float, default=0, help="random drop ratio")
    parser.add_argument('--gradually',  action='store_true', default=False)

    parser.add_argument('--drop_area_ratio', type=float, default=0.1)
    parser.add_argument('--drop_event_ratio', type=float, default=0.1)
    parser.add_argument('--slicing_time_window', type=float, default=500000)
    args = parser.parse_args()
    
    settings_filepath = args.settings_file
    if args.auto_resume:
        settings_filepath = os.path.join(args.save_dir, "settings.yaml")
        print("using stored config: {}".format(settings_filepath))
    settings = Settings(settings_filepath, generate_log=False)
    settings = utils.reinit_settings(settings, args)
    with open(settings_filepath, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)

    if not args.evaluate and not os.path.exists(args.save_dir) and args.save_dir:
        os.makedirs(args.save_dir)
        import shutil
        shutil.copy(settings_filepath, os.path.join(args.save_dir, "settings.yaml"))
    if args.save_dir:
        if not utils.check_trained_folder(args) and "test" not in args.save_dir:
            print("This folder contains trained model!")
            sys.exit(1)

    if args.data_path != "":
        cfg["dataset"]["dataset_path"] = args.data_path
    if args.generate_int_model:
        args.evaluate = True

    use_random = utils.check_random(settings, cfg)
    nr_input_channels = utils.get_input_channel(settings.event_representation)

    Dataset = utils.select_dataset(cfg["dataset"]["name"])
    MNIST = True if "MNIST" in cfg["dataset"]["name"] or "Poker" in cfg["dataset"]["name"] else False

    val_dataset = Dataset(cfg, mode="validation", shuffle=False, dataset_percentage=args.dataset_percentage,
                          min_event=args.min_event, slicing_time_window=args.slicing_time_window)
    if cfg["dataset"]["name"] == "IniRoshambo":
        nr_input_channels = 1

    nr_classes = val_dataset.nr_classes
    val_loader = Loader(val_dataset, batch_size=settings.batch_size, device=settings.gpu_device,
                        num_workers=settings.num_cpu_workers, pin_memory=False, shuffle=False)

    train_quant, base_model = False, False
    if "mink_mobilenetv2_int" in settings.model_name:
        model = MobileNetV2MEInt(num_classes=nr_classes, in_channels=nr_input_channels, width_mult=settings.width_mult,
                                 MNIST=MNIST)
        model.init_weights()
        utils.load_int_ckpt(model, args.load)
        args.evaluate = True
        print("You are loading integer model. Evaluation only.")
    elif settings.model_name == "base_mobilenetv2":
        base_model = True
        model = mobilenet_v2(num_classes=nr_classes, width_mult=settings.width_mult, MNIST=MNIST,
                             sample_channel=nr_input_channels, model_type=settings.model_type)
    elif 'mink_mobilenetv2' in settings.model_name:
        model = MobileNetV2ME(num_classes=nr_classes, in_channels=nr_input_channels, width_mult=settings.width_mult,
                              MNIST=MNIST, remove_depth=settings.remove_depth, drop_config=settings.drop_config,
                              relu=settings.relu_type, model_type=settings.model_type)
        if settings.model_name == 'mink_mobilenetv2_quant':
            train_quant = True
            assert args.load, "Please specify the path to the baseline model"
            model = utils.load_model(args.load, model)
            model = Q_MobileNetV2(model, nr_input_channels, settings.width_mult, nr_classes, conv1_bit=args.conv1_level,
                                  fix_BN_threshold=args.fixBN_ratio, MNIST=MNIST, shift_bit=args.shift_bit,
                                  drop_config=settings.drop_config, model_type=settings.model_type, bias_bit=args.bias_bit)
    elif 'mink_resnet18' in settings.model_name:
        model = resnet18(num_classes=nr_classes, input_dim=nr_input_channels)
        train_quant = False
    else:
        raise NotImplementedError

    param = sum(p.numel() for p in model.parameters())
    print("Number of ALL parameters: {}".format(param))
    param_noBN = sum(p.numel() for p in model.parameters() if len(p.shape) > 1)
    print("Number of parameters except BN: {}".format(param_noBN))
    model.to(settings.gpu_device)
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    # criterion = nn.CrossEntropyLoss()
    criterion = Loss(settings, args.lamb)

    # specify different optimizer to different training stage
    if not args.evaluate:
        if settings.optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=settings.init_lr)
        elif settings.optimizer == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                             lr=settings.init_lr,
                                             momentum=0.9,
                                             weight_decay=1e-4)
        else:
            raise NotImplementedError
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings.steps_lr, gamma=settings.factor_lr)
    else:
        optimizer = None
        scheduler = None

    start_epoch, best_acc1 = utils.load_checkpoint(args, model, optimizer)
    start_epoch += 1
    cudnn.benchmark = True

    use_abs_drop = utils.check_abs_sum_drop(settings)

    if args.evaluate:
        if not use_random:
            metrics, _ = validate(val_loader, model, criterion, settings.gpu_device, error_ana_path=args.ana_file,
                                  generate_int_model=args.generate_int_model, use_abs_drop=use_abs_drop, base_model=base_model)
            print(round(metrics[0],2), round(metrics[1],2), round(metrics[2],2), param, param_noBN)
            return metrics + [param, param_noBN]
        else:
            metrics, _ = utils.multiple_validate(validate, val_loader, model, criterion, settings.gpu_device,
                                                 args.ana_file, generate_int_model=args.generate_int_model,
                                                 use_abs_drop=use_abs_drop, times=1, base_model=base_model)
            print(round(metrics[0],2), round(metrics[1],2), round(metrics[2],2), param, param_noBN)
            return metrics + [param, param_noBN]

    train_dataset = Dataset(cfg, mode="training", dataset_percentage=args.dataset_percentage,
                            min_event=args.min_event, slicing_time_window=args.slicing_time_window)
    train_loader = Loader(train_dataset, batch_size=settings.batch_size, device=settings.gpu_device,
                          num_workers=settings.num_cpu_workers, pin_memory=False, shuffle=True)

    if args.gen_meta:
        sys.exit(0)

    if args.save_dir:
        train_log, test_log, msg_log, global_log = logger.handle_loggers(args)
        title = "Epoch\tPrec@1\tPrec@5\tloss"
        train_file = logger.open_with_title(train_log, title)
        test_file = logger.open_with_title(test_log, title)
        msg_file = open(msg_log, "a+")
        os.makedirs(os.path.join(args.save_dir, "hist"), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "confusion_matrix"), exist_ok=True)
    else:
        train_file, test_file, msg_file, global_log = None, None, None, None

    writer = None
    if use_tb and args.save_dir:
        writer = SummaryWriter(args.save_dir)

    best_metric = None

    for epoch in range(start_epoch, args.epochs):

        utils.update_epoch(model, args, epoch)

        if train_quant:
            unfreeze_model(model)
        batch_img = train(train_loader, model, criterion, settings.gpu_device, optimizer, writer, epoch, train_file,
                          base_model=base_model)
        if train_quant:
            freeze_model(model)
        if not use_random:
            metrics, matrix = validate(val_loader, model, criterion, settings.gpu_device, epoch, writer, test_file,
                                       args.ana_file, use_abs_drop=use_abs_drop, msg_log=msg_file, base_model=base_model)
        else:
            metrics, matrix = utils.multiple_validate(validate, val_loader, model, criterion, settings.gpu_device,
                                                      epoch, writer, test_file, args.ana_file, base_model=base_model,
                                                      use_abs_drop=use_abs_drop, times=5, msg_log=msg_file)
        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, "confusion_matrix", "{}.jpg".format(epoch)), matrix)
        scheduler.step()

        acc1 = metrics[0]
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        output_best = 'Best Acc@1: %.3f\n' % best_acc1
        print(output_best)

        if args.save_dir:
            utils.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': acc1,
            }, '%s/ckpt.pth.tar' % (args.save_dir), is_best, epoch/args.epochs)
        if is_best:
            best_metric = metrics
    if best_metric is not None:
        logger.log_global(best_metric, args, global_log)


def validate(val_loader, model, criterion, device, epoch=-1, writer=None, epoch_log=None, error_ana_path=None,
             generate_int_model=False, use_abs_drop=False, msg_log=None, base_model=False):
    val_loader_desc = tqdm.tqdm(val_loader, ascii=True, mininterval=5, total=len(val_loader))
    Recorder = logger.MetricRecorder("valid")
    model = model.eval()
    nr_classes = val_loader.loader.dataset.nr_classes
    val_confusion_matrix = np.zeros([nr_classes, nr_classes])

    if error_ana_path is not None:
        analyser = logger.ErrorAnalyser(error_ana_path)
    if use_abs_drop:
        prune_logger = logger.PruneRatioLogger()
    
    for i_batch, sample_batched in enumerate(val_loader_desc):

        with torch.no_grad():
            if base_model:
                model_output = model(sample_batched["histograms"].to(device))
            else:
                minknet_input = ME.SparseTensor(
                    coordinates=sample_batched["coordinates"], features=sample_batched["features"], device=device
                )
                model_output = model(minknet_input)
        labels = torch.tensor(sample_batched["labels"], dtype=torch.long).to(device)

        if model_output.shape[0] != len(sample_batched["labels"]):
            print("Error: batch size mismatch!")
            continue

        loss = criterion(model_output, labels, model)
        try:
            acc1, acc5 = utils.accuracy(model_output, labels, topk=(1, 5))
        except:
            acc1, acc5 = utils.accuracy(model_output, labels, topk=(1, 1))
        Recorder.update([acc1, acc5, loss], [], labels)

        val_loader_desc.set_description(
            'Valid: {epoch} | loss: {loss:.4f} | Top-1: {acc1:.2f} | Top-5: {acc5:.2f}'.
            format(epoch=epoch, loss=loss.item(), acc1=Recorder.get_metric("Top-1"), acc5=Recorder.get_metric("Top-5"))
        )

        if generate_int_model:
            int_model = MobileNetV2MEInt(num_classes=model.num_classes, in_channels=model.in_channels,
                                         width_mult=model.width_mult, MNIST=model.MNIST)
            int_model.init_weights(model)

            torch.save(int_model.state_dict(), "int_model.pth.tar")
            utils.calculate_weight_sparsity(int_model)
            print("Integer model generated!")
            sys.exit(1)

        if error_ana_path is not None:
            analyser.update(model_output, labels)

        if use_abs_drop:
            prune_logger.update(model)
        
        # Save validation statistics
        predicted_classes = model_output.argmax(1)
        np.add.at(val_confusion_matrix, (predicted_classes.data.cpu().numpy(), labels.data.cpu().numpy()), 1)

    Top1, Top5, Loss = Recorder.summary(epoch)
    val_confusion_matrix = val_confusion_matrix / (np.sum(val_confusion_matrix, axis=-1,
                                                                    keepdims=True) + 1e-9)
    plot_confusion_matrix = visualizations.visualizeConfusionMatrix(val_confusion_matrix)
    if epoch_log is not None:
        epoch_log.write("{}, {}, {}, {}\n".format(epoch, Top1, Top5, Loss))
    if error_ana_path is not None:
        analyser.finish()
    if use_abs_drop:
        prune_logger.release(msg_log, epoch)

    if writer is not None:
        writer.add_image('Validation/Confusion_Matrix', plot_confusion_matrix, epoch, dataformats='HWC')
        writer.add_scalar('Validation/Validation_Accuracy', Top1, epoch)
        writer.add_scalar('Validation/Validation_Loss', Loss, epoch)
    output = ('Validating Epoch {epoch}: Prec@1 {top1:.3f} Loss {loss:.5f} '.format(
        epoch=epoch, top1=Top1, loss=Loss))
    return [Top1, Top5, Loss], plot_confusion_matrix


def train(train_loader, model, criterion, device, optimizer, writer, epoch, epoch_log=None, base_model=False,
          msg_log=None):
    model = model.train()
    train_loader_desc = tqdm.tqdm(train_loader, ascii=True, mininterval=5, total=len(train_loader))
    Recorder = logger.MetricRecorder("train")

    for i_batch, sample_batched in enumerate(train_loader_desc):
        # _, labels, histogram = sample_batched
        
        optimizer.zero_grad()
        if base_model:
            model_output = model(sample_batched["histograms"].to(device))
        else:
            minknet_input = ME.SparseTensor(
                coordinates=sample_batched["coordinates"], features=sample_batched["features"], device=device
            )
            model_output = model(minknet_input)
        if model_output.shape[0] != len(sample_batched["labels"]):
            print("Error: batch size mismatch!")
            continue
        labels = torch.tensor(sample_batched["labels"], dtype=torch.long).to(device)

        loss = criterion(model_output, labels, model)
        if torch.isnan(loss):
            print("Removing all pruning batch")
            continue

        try:
            acc1, acc5 = utils.accuracy(model_output, labels, topk=(1, 5))
        except:
            acc1, acc5 = utils.accuracy(model_output, labels, topk=(1, 1))
        loss.backward()
        optimizer.step()
        Recorder.update([acc1, acc5, loss], [], labels)

        train_loader_desc.set_description(
            'Train: {epoch} | loss: {loss:.4f} | Top-1: {acc1:.2f} | Top-5: {acc5:.2f}'.
            format(epoch=epoch, loss=loss.item(), acc1=Recorder.get_metric("Top-1"), acc5=Recorder.get_metric("Top-5"))
        )

        # if i_batch == 0:
        #     # image = utils.histogram_visualize(histogram, model_input_size,
        #     #                                   [train_loader.loader.dataset.object_classes[idx] for idx in labels])
        #     # if writer is not None:
        #     #     writer.add_image('Training/Input Histogram', image, epoch, dataformats='HWC')

    Top1, Top5, Loss = Recorder.summary(epoch)
    if epoch_log is not None:
        epoch_log.write("{}, {}, {}, {}\n".format(epoch, Top1, Top5, Loss))
    if writer is not None:
        writer.add_scalar('Training/Training_Accuracy', Top1, epoch)
        writer.add_scalar('Training/Training_Loss', Loss, epoch)

    output = ('Training Epoch {epoch}: Prec@1 {top1:.3f} Loss {loss:.5f} '.format(
        epoch=epoch,
        top1=Top1,
        loss=Loss))
    print(output)
    return None
    # return image


if __name__ == '__main__':
    main()
