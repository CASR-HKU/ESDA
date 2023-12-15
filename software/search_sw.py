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
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Software searching')
    parser.add_argument('--num', '-n', help='Number of generating configs', type=int, default=10)
    for _ in range(parser.parse_args().num):
        print(generate_cfg())
