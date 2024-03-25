import os

class MetricRecorder:
    def __init__(self, log_file=""):
        self.metric_names = ["tr_p10_acc", "val_p10_acc", "train_loss", "val_loss", "tr_p5_acc",
                             "val_p5_acc", "tr_p15_acc", "val_p15_acc", "tr_p_error", "val_p_error"]
        self.metrics = {name: [] for name in self.metric_names}
        self.metrics_direction = ["h", "h", "l", "l", "h", "h", "h", "h", "l", "l"]
        best_init_value = {"l": float("inf"), "h": 0}
        self.metrics_best = {name: best_init_value[direction]
                             for name, direction in zip(self.metric_names, self.metrics_direction)}
        if log_file:
            self.log_file = open(log_file, "w")
            self.log_file.write("epoch,")
            self.log_file.write(",".join(self.metric_names))
            self.log_file.write("\n")

    def update(self, metric_dict, epoch):
        for name in self.metric_names:
            self.metrics[name].append(metric_dict[name])
            if self.better(metric_dict[name], self.metrics_best[name], self.metrics_direction[self.metric_names.index(name)]):
                self.metrics_best[name] = metric_dict[name]
        if self.log_file:
            self.log_file.write(f"{epoch},")
            self.log_file.write(",".join([str(round(metric_dict[name], 2)) for name in self.metric_names]))
            self.log_file.write("\n")

    def better(self, current, prev, direction):
        if direction == "l":
            return current < prev
        else:
            return current > prev

    def get_best(self):
        return [self.metrics_best[name] for name in self.metric_names]

    def transform_metric(self, train_loss, val_loss, train_metric, val_metric):
        metrics = {}
        metrics["train_loss"] = round(train_loss, 4)
        metrics["val_loss"] = round(val_loss, 4)
        metrics["tr_p5_acc"] = round(train_metric["tr_p_acc"]["tr_p5_acc"], 4)
        metrics["tr_p10_acc"] = round(train_metric["tr_p_acc"]["tr_p10_acc"], 4)
        metrics["tr_p15_acc"] = round(train_metric["tr_p_acc"]["tr_p15_acc"], 4)
        metrics["tr_p_error"] = round(train_metric['tr_p_error']["tr_p_error"], 2)
        metrics["val_p5_acc"] = round(val_metric["val_p_acc"]["val_p5_acc"], 4)
        metrics["val_p10_acc"] = round(val_metric["val_p_acc"]["val_p10_acc"], 4)
        metrics["val_p15_acc"] = round(val_metric["val_p_acc"]["val_p15_acc"], 4)
        metrics["val_p_error"] = round(val_metric['val_p_error']["val_p_error"], 2)
        return metrics


class ExpRecorder:
    def __init__(self, file_name):
        if os.path.exists(file_name):
            self.file = open(file_name, "a+")
        else:
            self.file = open(file_name, "w")
            self.file.write("exp_id, flops, params, ")
            self.file.write("train_p5_acc, val_p5_acc, train_p10_acc, val_p10_acc, train_loss, val_loss, tr_p_error, "
                            "val_p_error\n")

    def update(self, args, eff, metrics):
        flops = round(eff[0], 2)
        params = eff[1]
        exp_id = args.mlflow_path.split("/")[-1]
        train_p10_acc, val_p10_acc, train_loss, val_loss, tr_p_error, val_p_error = \
            metrics[0], metrics[1], metrics[2], metrics[3], metrics[-2], metrics[-1]
        train_p5_acc, val_p5_acc = metrics[4], metrics[5]
        self.file.write(f"{exp_id}, {flops}, {params}, ")
        self.file.write(f"{train_p5_acc}, {val_p5_acc}, ")
        self.file.write(f"{train_p10_acc}, {val_p10_acc}, {train_loss}, {val_loss}, {tr_p_error}, {val_p_error}\n")
