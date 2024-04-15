import torch
import os
from utils.metrics import p_acc, p_acc_wo_closed_eye, px_euclidean_dist, get_prediction
import tqdm


def gen_npy(model, val_loader, criterion, args, epoch=0):
    model.eval()

    val_loader_desc = tqdm.tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for inputs, targets, _ in val_loader_desc:
            # inputs = inputs.permute(0, 1, 2, 4, 3)
            model(inputs.to(args.device), int_folder=args.int_folder)
            return

        
def gen_json(model, val_loader, args, int_folder="", ):
    val_loader_desc = tqdm.tqdm(val_loader, ascii=True, mininterval=5, total=len(val_loader))
    model = model.eval()

    dataset_name = "EyeTracking"
    size = (args.sensor_height*args.spatial_factor, args.sensor_width*args.spatial_factor)
    model.json_file = os.path.join(int_folder, "model.json")

    for i_batch, (inputs, target, _) in enumerate(val_loader_desc):
        with torch.no_grad():
            model(inputs, dataset_name, size)
            return


def train_epoch(model, train_loader, criterion, optimizer, args, epoch=0):
    model.train()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all = {f'error_all':0}  # averaged euclidean distance
    total_samples_all, total_sample_p_error_all  = 0, 0

    train_loader_desc = tqdm.tqdm(train_loader, desc="Training")
    for inputs, targets, masked_lengths in train_loader_desc:
    # for inputs, targets in train_loader:
        optimizer.optimizer.zero_grad()
        outputs = model(inputs.to(args.device))
        #taking only the last frame's label, and first two dim are coordinate, last is open or close so discarded
        if model.use_heatmap:
            targets, hms = targets
            targets, hms = targets.to(args.device), hms.to(args.device)
            loss = criterion(outputs, hms)
            b, t, w, h = hms.shape
            output_pixels, _ = get_prediction(outputs.reshape(1, b*t, w, h))
            output_pixels[:, :, 0] /= (args.sensor_width*args.spatial_factor)
            output_pixels[:, :, 1] /= (args.sensor_height*args.spatial_factor)
            # calculate pixel tolerated accuracy
            p_corr, batch_size = p_acc(targets[:, :, :2].reshape(-1, 2), output_pixels,
                                       width_scale=args.sensor_width*args.spatial_factor,
                                       height_scale=args.sensor_height*args.spatial_factor,
                                       pixel_tolerances=args.pixel_tolerances)
            p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :], output_pixels,
                                                               width_scale=args.sensor_width * args.spatial_factor,
                                                               height_scale=args.sensor_height * args.spatial_factor)
        else:
            targets = targets.to(args.device)
            if masked_lengths.sum() > 0:
                for idx, masked_length in enumerate(masked_lengths):
                    outputs[idx, :masked_length] = 0
            loss = criterion(outputs, targets[:, :, :2])
            # calculate pixel tolerated accuracy
            p_corr, batch_size = p_acc(targets[:, -1, :2], outputs[:, -1, :],
                                       width_scale=args.sensor_width*args.spatial_factor,
                                       height_scale=args.sensor_height*args.spatial_factor,
                                       pixel_tolerances=args.pixel_tolerances)
            p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :], outputs[:, :, :],
                                                               width_scale=args.sensor_width * args.spatial_factor,
                                                               height_scale=args.sensor_height * args.spatial_factor)

        loss.backward()
        optimizer.optimizer.step()
        total_loss += loss.item()

        total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
        total_samples_all += batch_size

        # calculate averaged euclidean distance

        total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
        total_sample_p_error_all += bs_times_seqlen

        train_loader_desc.set_description(
            'Train: {epoch} | loss: {loss:.4f} | tp10: {tp10:.4f} | tp5: {tp5:.4f} | dist: {dist:.4f}'.
            format(epoch=epoch, loss=total_loss/total_samples_all,
                   tp10=total_p_corr_all[f'p10_all']/total_samples_all,
                   tp5=total_p_corr_all[f'p5_all'] / total_samples_all,
                   dist=total_p_error_all[f'error_all']/total_sample_p_error_all)
        )
    metrics = {'tr_p_acc': {f'tr_p{k}_acc': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
               'tr_p_error': {f'tr_p_error': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}
    
    return model, total_loss / len(train_loader), metrics


def validate_epoch(model, val_loader, criterion, args, epoch=0):
    model.eval()
    total_loss = 0.0
    total_p_corr_all = {f'p{p}_all':0 for p in args.pixel_tolerances}
    total_p_error_all  = {f'error_all':0}
    total_samples_all, total_sample_p_error_all  = 0, 0

    val_loader_desc = tqdm.tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for inputs, targets, _ in val_loader_desc:
            if args.debug:
                print("{}, {}, {}".format(inputs.sum(), inputs.mean(), inputs.var()))
            outputs = model(inputs.to(args.device))
            if model.use_heatmap:
                targets, hms = targets
                targets, hms = targets.to(args.device), hms.to(args.device)
                loss = criterion(outputs, hms)
                b, t, w, h = hms.shape
                output_pixels, _ = get_prediction(outputs.reshape(1, b * t, w, h))
                output_pixels[:, :, 0] /= (args.sensor_width * args.spatial_factor)
                output_pixels[:, :, 1] /= (args.sensor_height * args.spatial_factor)
                # calculate pixel tolerated accuracy
                p_corr, batch_size = p_acc(targets[:, :, :2].reshape(-1, 2), output_pixels,
                                           width_scale=args.sensor_width * args.spatial_factor,
                                           height_scale=args.sensor_height * args.spatial_factor,
                                           pixel_tolerances=args.pixel_tolerances)
                p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :], output_pixels,
                                                                   width_scale=args.sensor_width * args.spatial_factor,
                                                                   height_scale=args.sensor_height * args.spatial_factor)
            else:
                targets = targets.to(args.device)
                loss = criterion(outputs, targets[:, :, :2])
                # calculate pixel tolerated accuracy
                p_corr, batch_size = p_acc(targets[:, -1, :2], outputs[:, -1, :],
                                           width_scale=args.sensor_width * args.spatial_factor,
                                           height_scale=args.sensor_height * args.spatial_factor,
                                           pixel_tolerances=args.pixel_tolerances)
                p_error_total, bs_times_seqlen = px_euclidean_dist(targets[:, :, :], outputs[:, :, :],
                                                                   width_scale=args.sensor_width * args.spatial_factor,
                                                                   height_scale=args.sensor_height * args.spatial_factor)
            total_loss += loss.item()

            total_p_corr_all = {f'p{k}_all': (total_p_corr_all[f'p{k}_all'] + p_corr[f'p{k}']).item() for k in args.pixel_tolerances}
            total_samples_all += batch_size

            total_p_error_all = {f'error_all': (total_p_error_all[f'error_all'] + p_error_total).item()}
            total_sample_p_error_all += bs_times_seqlen

            val_loader_desc.set_description(
                'Valid: {epoch} | loss: {loss:.4f} | tp10: {tp10:.4f} | tp5: {tp5:.4f} | dist: {dist:.4f}'.
                format(epoch=epoch, loss=total_loss / total_samples_all,
                       tp10=total_p_corr_all[f'p10_all'] / total_samples_all,
                       tp5=total_p_corr_all[f'p5_all'] / total_samples_all,
                       dist=total_p_error_all[f'error_all'] / total_sample_p_error_all)
            )

    metrics = {'val_p_acc': {f'val_p{k}_acc': (total_p_corr_all[f'p{k}_all']/total_samples_all) for k in args.pixel_tolerances},
                'val_p_error': {f'val_p_error': (total_p_error_all[f'error_all']/total_sample_p_error_all)}}
    
    return total_loss / len(val_loader), metrics


def top_k_checkpoints(args, artifact_uri):
    """
    only save the top k model checkpoints with the lowest validation loss.
    """
    # list all files ends with .pth in artifact_uri
    model_checkpoints = [f for f in os.listdir(artifact_uri) if f.endswith(".pth")]

    # but only save at most args.save_k_best models checkpoints
    if len(model_checkpoints) > args.save_k_best:
        # sort all model checkpoints by validation loss in ascending order
        model_checkpoints = sorted([f for f in os.listdir(artifact_uri) if f.startswith("model_best_ep")], \
                                    key=lambda x: float(x.split("_")[-1][:-4]))
        # delete the model checkpoint with the largest validation loss
        os.remove(os.path.join(artifact_uri, model_checkpoints[-1]))


