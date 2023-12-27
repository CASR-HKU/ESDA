import json

def NMNIST_L(remove_depth=0):
    if remove_depth == 0:
        depth_cfg = [1, 3, 4, 3, 2]
    elif remove_depth == 1:
        depth_cfg = [1, 3, 4, 2, 2]
    elif remove_depth == 2:
        depth_cfg = [1, 2, 4, 2, 2]
    elif remove_depth == 3:
        depth_cfg = [1, 2, 3, 2, 2]
    elif remove_depth == 4:
        depth_cfg = [1, 2, 2, 2, 2]
    elif remove_depth == 5:
        depth_cfg = [1, 2, 2, 2, 1]
    elif remove_depth == 6:
        depth_cfg = [1, 2, 2, 1, 1]
    elif remove_depth == 7:
        depth_cfg = [1, 1, 2, 1, 1]
    elif remove_depth == 8:
        depth_cfg = [1, 1, 1, 1, 1]
    elif remove_depth == 9:
        depth_cfg = [1, 1, 1, 1, 0]
    elif remove_depth == 10:
        depth_cfg = [1, 1, 0, 1, 0]
    else:
        raise ValueError("You are removing too many layers!")
    return depth_cfg


def NMNIST_base(remove_depth=0):
    if remove_depth == 0:
        depth_cfg = [1, 2, 4, 3, 1]
    elif remove_depth == 1:
        depth_cfg = [1, 2, 4, 2, 1]
    elif remove_depth == 2:
        depth_cfg = [1, 2, 3, 2, 1]
    elif remove_depth == 3:
        depth_cfg = [1, 2, 2, 2, 1]
    elif remove_depth == 4:
        depth_cfg = [1, 2, 2, 1, 1]
    elif remove_depth == 5:
        depth_cfg = [1, 2, 1, 1, 1]
    elif remove_depth == 6:
        depth_cfg = [1, 1, 1, 1, 1]
    elif remove_depth == 7:
        depth_cfg = [1, 1, 1, 1, 0]
    elif remove_depth == 8:
        depth_cfg = [1, 1, 0, 1, 0]
    else:
        raise ValueError("You are removing too many layers!")
    return depth_cfg


def get_roshambo_config(remove_depth=0, model_type="base", drop_config={}):
    inverted_residual_setting = [
        # t, c, s
        [1, 16, 1, 1, 0],
        [4, 24, 1, 2, 0],
        [4, 32, 2, 2, 0],
        [4, 64, 2, 1, 0],
        [4, 72, 1, 2, 0],
    ]

    return inverted_residual_setting


def get_MNIST_config(remove_depth=0, model_type="base", drop_config={}):
    if model_type == "base":
        depth_cfg = NMNIST_base(remove_depth)
    elif model_type == "L":
        depth_cfg = NMNIST_L(remove_depth)
    else:
        raise ValueError("model type should be either base or L")
    inverted_residual_setting = [
        # t, c, s
        [1, 16, 1, 1, 0],
        [6, 24, 2, 2, 0],
        [6, 32, 4, 1, 0],
        [6, 64, 3, 2, 0],
        [6, 96, 1, 1, 0],
    ]

    for i in range(len(depth_cfg)):
        inverted_residual_setting[i][2] = depth_cfg[i]
    return inverted_residual_setting


def get_config(model_type="base"):
    input_channel, output_channel = 32, 1280

    if model_type == "base":
        inverted_residual_setting = [
            # t, c, n, s, drop
            [1, 16, 1, 1, 0],
            [6, 24, 2, 2, 0],
            [6, 32, 3, 2, 0],
            [6, 64, 4, 2, 0],
            [6, 96, 3, 1, 0],
            [6, 160, 3, 2, 0],
            [6, 320, 1, 1, 0],
        ]
    else:
        inverted_residual_setting, input_channel, output_channel = load_json(model_type)

    return inverted_residual_setting, input_channel, output_channel


def generate_cfg():
    import random
    channels_candidate = [16, 24, 32, 48, 64, 72, 96, 128, 144, 160, 196, 224, 256, 320]
    stage_numbers = [4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9]
    repeat_times = [1, 1, 1, 2, 2, 2, 3, 3, 4]
    expand_ratios = [1, 2, 2, 4, 4, 6, 6, 8]
    downsample_nums = 4

    stages_num = random.choice(stage_numbers)
    channels = random.sample(channels_candidate, stages_num)
    channels.sort()
    stage_repeats = [random.choice(repeat_times) for _ in channels]
    expand_ratios = [random.choice(expand_ratios) for _ in channels]
    downsample_stages = random.sample(range(stages_num), downsample_nums)
    inverted_residual_setting = []
    for idx, (c, r, e) in enumerate(zip(channels, stage_repeats, expand_ratios)):
        s = 2 if idx in downsample_stages else 1
        inverted_residual_setting.append([e, c, r, s, 0])
    return inverted_residual_setting


def load_json(json_path):
    with open(json_path, "r") as f:
        settings = json.loads(f.read())
        inverted_residual_setting = settings["settings"]
    first_channel = settings["layers"][0]["channels"][1]
    final_layer_settings = settings["layers"][-1] if settings["layers"][-1]["type"] == "conv" else settings["layers"][-2]
    final_channel = final_layer_settings["channels"][1]
    return inverted_residual_setting, first_channel, final_channel


if __name__ == '__main__':
    for _ in range(10):
        print(generate_cfg())
