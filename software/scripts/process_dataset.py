import sys
import os
import numpy as np
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from PyAedatTools.ImportAedat import ImportAedat
import extractdata_uti as uti
import argparse
import pickle


# width = int(x.max()) + 1
# height = int(y.max()) + 1
def denoise(events, height, width, filter_time=20000):
    events_copy = np.zeros_like(events)
    copy_index = 0
    timestamp_memory = np.zeros((width, height)) + filter_time

    for event in events:
        x = event[0]
        y = event[1]
        t = event[2]
        timestamp_memory[x, y] = t + filter_time
        if (
                (x > 0 and timestamp_memory[x - 1, y] > t)
                or (x < width - 1 and timestamp_memory[x + 1, y] > t)
                or (y > 0 and timestamp_memory[x, y - 1] > t)
                or (y < height - 1 and timestamp_memory[x, y + 1] > t)
        ):
            events_copy[copy_index] = event
            copy_index += 1

    return events_copy[:copy_index]


def generate_test_data(WINDOW_SIZE, STEP_SIZE, SEQ_LEN, NUM_POINTS, DENOISE=False, FILTER_TIME=20000):
    EXPORT_PATH = OUT_TEST

    print('Data will save to', EXPORT_PATH)
    DATA_PER_FILE = 4000000

    test_data, test_timelabel = uti.get_file_list(TEST_FILE)
    NUM_TRAIN_FILE = len(test_data)
    # os.remove(os.path.join(EXPORT_PATH,'test_0.h5'))
    # os.remove(os.path.join(EXPORT_PATH,'test_1.h5'))

    row_count = 0
    exp_count = 0

    for j in range(NUM_TRAIN_FILE):  # NUM_TRAIN_FILE
        output_name = "test_" + str(j) + ".pkl"
        out_dir = os.path.join(EXPORT_PATH, output_name)

        data = []
        label = []

        # --------Get time lable for each class in each video----------------
        print('----------Processing File No.', j, '------------')
        # print('Processing Train Data File: ',test_data[j])
        # print('Reading Train Lable File: ',test_timelabel[j])
        class_label = []
        class_start_timelabel = []
        class_end_timelabel = []
        with open(os.path.join(DATA_PATH, test_timelabel[j])) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                class_label.append(row[0])
                class_start_timelabel.append(row[1])
                class_end_timelabel.append(row[2])
        del class_label[0]
        del class_start_timelabel[0]
        del class_end_timelabel[0]
        class_label = list(map(int, class_label))
        class_start_timelabel = list(map(int, class_start_timelabel))
        class_end_timelabel = list(map(int, class_end_timelabel))

        # -------------Extract raw data (timestep,x,y)-----------------------
        aedat = {}
        aedat['importParams'] = {}
        aedat['importParams']['filePath'] = os.path.join(DATA_PATH, test_data[j])
        aedat = ImportAedat(aedat)
        timestep = np.array(aedat['data']['polarity']['timeStamp']).tolist()

        # ---------Extract each class from video------------------------------
        class_start_index, class_end_index = uti.get_class_index(timestep, class_start_timelabel, class_end_timelabel)

        # ---------Extract data by sliding window for each class--------------
        for i in range(len(class_label)):
            data_temp = []
            label_temp = []

            if class_label[i] > NUM_CLASSES:
                continue

            # print('EXtraciting class-',class_label[i]-1)

            class_timestep = timestep[class_start_index[i]:class_end_index[i]]
            class_events = np.zeros(shape=(len(class_timestep), 4), dtype=np.int32)
            class_events[:, 0] = class_timestep
            class_events[:, 1] = aedat['data']['polarity']['x'][class_start_index[i]:class_end_index[i]]
            class_events[:, 2] = aedat['data']['polarity']['y'][class_start_index[i]:class_end_index[i]]
            class_events[:, 3] = aedat['data']['polarity']['polarity'][class_start_index[i]:class_end_index[i]]

            win_start_index, win_end_index = uti.get_window_index(class_timestep, class_timestep[0],
                                                                  stepsize=STEP_SIZE * 1000000,
                                                                  windowsize=WINDOW_SIZE * 1000000)

            NUM_WINDOWS = len(win_start_index)

            for n in range(NUM_WINDOWS):  # NUM_WINDOWS

                window_events = class_events[win_start_index[n]:win_end_index[n], :].copy()

                # -------------Downsample---------------------------------
                extracted_events = window_events  # uti.shuffle_downsample(window_events,NUM_POINTS)

                # ------------Normalize Data------------------------------
                extracted_events[:, 0] = extracted_events[:, 0] - extracted_events[:, 0].min(axis=0)

                # ------------Arrange Data by Timestep-------------------------

                t = extracted_events[:, 0]
                x = extracted_events[:, 1]
                y = extracted_events[:, 2]
                p = extracted_events[:, 3]
                # print(extracted_events.shape)
                height = int(y.max()) + 1
                width = int(x.max()) + 1
                events_normed = np.vstack([x, y, t, p]).T
                # print(events_normed.shape)
                if DENOISE:
                    events_denoised = denoise(events_normed, height, width, FILTER_TIME)
                else:
                    events_denoised = events_normed
                print("before denoised:{} after:{}".format(events_normed.shape[0], events_denoised.shape[0]))
                data.append(events_denoised)
                label.append(class_label[i] - 1)
                # if (n + 1)%SEQ_LEN == 0:
                #     data.append(data_temp)
                #     label.append(label_temp)
                #     label_temp = []
                #     data_temp =[]

        # ------------------------Shuffle and Reshape Data-------------------------
        # data = np.array(data)
        label = np.array(label)

        # print(label.shape)

        # ------------------------Store Data as HDF5 file-------------------------

        packed = {}
        packed['data'] = data
        packed['label'] = label
        file = open(out_dir, 'wb')
        pickle.dump(packed, file)
        file.close()


def generate_train_data(WINDOW_SIZE, STEP_SIZE, SEQ_LEN, NUM_POINTS, DENOISE=False, FILTER_TIME=20000):
    EXPORT_PATH = OUT_TRAIN
    print('Data will save to', EXPORT_PATH)
    train_data, train_timelabel = uti.get_file_list(TRAIN_FILE)
    NUM_TRAIN_FILE = len(train_data)

    row_count = 0
    exp_count = 0

    for j in range(NUM_TRAIN_FILE):  # NUM_TRAIN_FILE
        output_name = "train_" + str(j) + ".pkl"
        out_dir = os.path.join(EXPORT_PATH, output_name)
        data = []
        label = []
        # --------Get time lable for each class in each video----------------
        # print('----------Processing File No.',j, '------------')
        # print('Processing Train Data File: ',train_data[j])
        # print('Reading Train Lable File: ',train_timelabel[j])
        class_label = []
        class_start_timelabel = []
        class_end_timelabel = []
        with open(os.path.join(DATA_PATH, train_timelabel[j])) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                class_label.append(row[0])
                class_start_timelabel.append(row[1])
                class_end_timelabel.append(row[2])
        del class_label[0]
        del class_start_timelabel[0]
        del class_end_timelabel[0]
        class_label = list(map(int, class_label))
        class_start_timelabel = list(map(int, class_start_timelabel))
        class_end_timelabel = list(map(int, class_end_timelabel))

        # -------------Extract raw data (timestep,x,y)-----------------------
        aedat = {}
        aedat['importParams'] = {}
        aedat['importParams']['filePath'] = os.path.join(DATA_PATH, train_data[j])
        aedat = ImportAedat(aedat)
        timestep = np.array(aedat['data']['polarity']['timeStamp']).tolist()

        # ---------Extract each class from video------------------------------

        class_start_index, class_end_index = uti.get_class_index(timestep, class_start_timelabel, class_end_timelabel)

        # ---------Extract data by sliding window for each class--------------
        for i in range(len(class_label)):
            data_temp = []
            label_temp = []
            if class_label[i] > NUM_CLASSES:
                continue
            # print('EXtraciting class-',class_label[i]-1)

            class_timestep = timestep[class_start_index[i]:class_end_index[i]]
            class_events = np.zeros(shape=(len(class_timestep), 4), dtype=np.int32)
            class_events[:, 0] = class_timestep
            class_events[:, 1] = aedat['data']['polarity']['x'][class_start_index[i]:class_end_index[i]]
            class_events[:, 2] = aedat['data']['polarity']['y'][class_start_index[i]:class_end_index[i]]
            class_events[:, 3] = aedat['data']['polarity']['polarity'][class_start_index[i]:class_end_index[i]]

            win_start_index, win_end_index = uti.get_window_index(class_timestep, class_timestep[0],
                                                                  stepsize=STEP_SIZE * 1000000,
                                                                  windowsize=WINDOW_SIZE * 1000000)

            NUM_WINDOWS = len(win_start_index)

            for n in range(NUM_WINDOWS):  # NUM_WINDOWS

                window_events = class_events[win_start_index[n]:win_end_index[n], :].copy()

                # window_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                # print(window_events.shape)
                # -------------Downsample---------------------------------
                extracted_events = window_events  # uti.shuffle_downsample(window_events,NUM_POINTS)

                # ------------Normalize Data------------------------------
                extracted_events[:, 0] = extracted_events[:, 0] - extracted_events[:, 0].min(axis=0)
                # events_normed = extracted_events / extracted_events.max(axis=0)
                # events_normed[:,1] = extracted_events[:,1] / 127
                # events_normed[:,2] = extracted_events[:,2] / 127
                # ------------Append data---------------------------------

                t = extracted_events[:, 0]
                x = extracted_events[:, 1]
                y = extracted_events[:, 2]
                p = extracted_events[:, 3]
                events_normed = np.vstack([x, y, t, p]).T
                height = int(y.max()) + 1
                width = int(x.max()) + 1
                if DENOISE:
                    events_denoised = denoise(events_normed, height, width, FILTER_TIME)
                else:
                    events_denoised = events_normed
                print("before denoised:{} after:{}".format(events_normed.shape[0], events_denoised.shape[0]))
                data.append(events_denoised)
                label.append(class_label[i] - 1)

        # ------------------------Shuffle and Reshape Data-------------------------
        # data = np.array(data)
        label = np.array(label)

        idx_out = np.arange(label.shape[0])
        np.random.shuffle(idx_out)
        data = [data[i] for i in idx_out]
        label = label[idx_out]
        packed = {}
        packed['data'] = data
        packed['label'] = label
        file = open(out_dir, 'wb')
        pickle.dump(packed, file)
        file.close()


if __name__ == "__main__":

    WINDOW_SIZE = 0.5
    STEP_SIZE = 0.25
    SEQ_LEN = 1
    NUM_POINTS = 1024

    NUM_CLASSES = 10
    DATA_PATH = '/vol/datastore/gyz/event_dataset/DvsGesture'
    TRAIN_FILE = os.path.join(DATA_PATH, 'trials_to_train.txt')
    TEST_FILE = os.path.join(DATA_PATH, 'trials_to_test.txt')

    SAVE_PATH = BASE_DIR
    DATA_PER_FILE = 4000

    argparse = argparse.ArgumentParser()
    # store ture for denoise
    argparse.add_argument('--denoise', action='store_true', default=False)
    argparse.add_argument('--filter_time', type=int, default=20000)
    args = argparse.parse_args()

    if args.denoise:
        base_dir = '/vol/datastore/gyz/event_dataset'
        dir = 'dvs_gesture_clip_denoise_{}'.format(args.filter_time)
        dir = os.path.join(base_dir, dir)
        OUT_TEST = os.path.join(dir, 'test')
        OUT_TRAIN = os.path.join(dir, 'train')
        # OUT_TEST = '/dvs_gesture_clip_denoise_20000/test'
        # OUT_TRAIN = '/vol/datastore/gyz/event_dataset/dvs_gesture_clip_denoise_20000/train'
    else:
        OUT_TEST = '/vol/datastore/gyz/event_dataset/dvs_gesture_clip/test'
        OUT_TRAIN = '/vol/datastore/gyz/event_dataset/dvs_gesture_clip/train'

    if not os.path.exists(OUT_TEST):
        os.makedirs(OUT_TEST)

    if not os.path.exists(OUT_TRAIN):
        os.makedirs(OUT_TRAIN)

    generate_train_data(WINDOW_SIZE, STEP_SIZE, SEQ_LEN, NUM_POINTS, DENOISE=args.denoise, FILTER_TIME=args.filter_time)
    generate_test_data(WINDOW_SIZE, STEP_SIZE, SEQ_LEN, NUM_POINTS, DENOISE=args.denoise, FILTER_TIME=args.filter_time)