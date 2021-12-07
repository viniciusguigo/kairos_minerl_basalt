""" compile_labels.py

Compiles all available labels in a single file.

"""
import os, sys
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def load_original_dataset(labels_dataset_addr, relative_label_addr, dataset_addr):
    if dataset_addr is None:
        # Parse original dataset address from labels dataset address
        relative_label_addr_len = len(relative_label_addr)+1
        dataset_addr = labels_dataset_addr[relative_label_addr_len:]
        # print(f'Loading data from original dataset address: {dataset_addr}')

    # Load rendered.npz data
    data_dict = dict(np.load(os.path.join(dataset_addr, 'rendered.npz')))

    # Load metadata
    with open(os.path.join(dataset_addr, 'metadata.json')) as f:
        metadata_dict = json.load(f)

    return data_dict, metadata_dict    


def load_actions(labels_dataset_addr, relative_label_addr, dataset_addr=None):
    # Load original dataset
    data_dict, metadata_dict = load_original_dataset(labels_dataset_addr, relative_label_addr, dataset_addr)

    # Parse actions 
        # [0] "attack"
        # [1] "back"
        # [2] "equip"
        # [3] "forward"
        # [4] "jump"
        # [5] "left"
        # [6] "right"
        # [7] "sneak"
        # [8] "sprint"
        # [9] "use"
        # [10] "camera_up_down"
        # [11] "camera_right_left"
    actions = np.vstack((
        data_dict['action$attack'], data_dict['action$back'], data_dict['action$equip'],
        data_dict['action$forward'], data_dict['action$jump'], data_dict['action$left'],
        data_dict['action$right'], data_dict['action$sneak'], data_dict['action$sprint'],
        data_dict['action$use'], data_dict['action$camera'][:,0], data_dict['action$camera'][:,1]
    ))
    actions = actions.transpose()

    # There are more video frames than actions
    # ASSUMPTION: the initial video frames are when the minecraft is still loading,
    # so all actions are zero
    diff_frame_count = metadata_dict['true_video_frame_count']-metadata_dict['duration_steps']
    action_padding_template = ['0', '0', 'none', '0', '0', '0', '0', '0', '0', '0', '0.0', '0.0']
    action_padding = np.full((diff_frame_count,12), action_padding_template)
    actions = np.vstack((action_padding, actions))

    return actions


def print_summary(all_labels_np):
    '''Prints summary of labeled data.'''
    num_labels = all_labels_np.shape[0]
    print(f'Images labeled: {num_labels} images')
    for i in range(all_labels_np.shape[1]):
        labels_per_class = all_labels_np[all_labels_np[:,i] == 1]
        num_labels_per_class = labels_per_class.shape[0]
        print(f'  Labels for class {i}: {num_labels_per_class} ({100*num_labels_per_class/num_labels:.3f} %)')


def load_training_data(data_paths):
    data_images = None
    data_labels = None
    num_classes = 0

    for data_path in data_paths:
        data_images_for_directory = np.load(os.path.join(data_path, "images.npy"))
        data_labels_for_directory = np.load(os.path.join(data_path, "labels.npy"))
        if data_labels_for_directory.shape[0] != data_images_for_directory.shape[0]:
            raise ValueError("Training data not the same length: {} vs. {}".format(data_labels_for_directory.shape[0],
                                                                                   data_images_for_directory.shape[0]))

        if data_labels is None:
            data_labels = data_labels_for_directory
        else:
            if data_labels.shape[1] != data_labels_for_directory.shape[1]:
                raise ValueError(
                    "Data labels shape mismatch: {} vs. {}".format(data_labels.shape, data_labels_for_directory.shape))
            data_labels = np.concatenate([data_labels, data_labels_for_directory], axis=0)

        if data_images is None:
            data_images = data_images_for_directory
        else:
            if data_images.shape[1:] != data_images_for_directory.shape[1:]:
                raise ValueError(
                    "Data images shape mismatch: {} vs. {}".format(data_images.shape, data_images_for_directory.shape))
            data_images = np.concatenate([data_images, data_images_for_directory], axis=0)

    columns_all_zero = np.argwhere(np.all(data_labels[..., :] == 0, axis=0)).flatten()
    columns_not_all_zero = np.argwhere(np.any(data_labels[..., :] != 0, axis=0)).flatten()

    data_label_names = [
        'No Labels',
        'Has Cave',
        'Inside Cave',
        'Danger Ahead',
        'Has Mountain',
        'Facing Wall',
        'At the top of a waterfall',
        'Good view of waterfall',
        'Good view of pen',
        'Good view of house',
        'Has animals',
        'Has open space',
        'Animals inside pen',
    ]
    # print(columns_all_zero)
    # print(columns_not_all_zero)

    if columns_all_zero.shape[0] > 0:
        print("Labels missing: {}".format(data_label_names[columns_all_zero]))
    if columns_not_all_zero.shape[0] == 0:
        raise ValueError("Labels are all 0. Check data")
    # print("Labels contained: {}".format(data_label_names[columns_not_all_zero]))

    print("Labels included in dataset")
    label_indices = [(key, val) for key, val in enumerate(data_label_names) if key in set(columns_not_all_zero)]
    for classifier_index, (global_index, label_name) in enumerate(label_indices):
        print("Name: {}, Classifier Index: {}, Global Index: {}".format(label_name, classifier_index, global_index))

    data_labels = np.delete(data_labels, columns_all_zero, axis=1)
    data_labels = data_labels.astype(np.float32)
    num_classes = columns_not_all_zero.shape[0]
    data_images = data_images.transpose(0, 3, 1, 2)

    return data_images, data_labels, num_classes


def split_data(data_images, data_labels, train_size=0.75, val_size=0.15, test_size=0.1):
    if train_size + val_size + test_size != 1.0:
        raise ValueError("Percentages must sum to 1 ({} + {} + {} != 1.0)".format(train_size, val_size, test_size))
    # split into training and validation
    x_train, x_val, y_train, y_val = train_test_split(data_images, data_labels, test_size=val_size, random_state=2, stratify=data_labels)
    test_size = 1.0 - train_size / (train_size + test_size)  # 1 - 0.75 / (0.75+0.1)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=1, stratify=y_train)

    print("data_images: {}".format(data_images.shape))
    print("data_labels: {}".format(data_labels.shape))
    print("x_train: {}".format(x_train.shape))
    print("y_train: {}".format(y_train.shape))
    print("x_test: {}".format(x_test.shape))
    print("y_test: {}".format(y_test.shape))
    print("x_val: {}".format(x_val.shape))
    print("y_val: {}".format(y_val.shape))

    return x_train, y_train, x_val, y_val, x_test, y_test


def main():
    MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
    relative_label_addr = 'labels'

    # Find all labelled tasks
    label_tasks = glob.glob(os.path.join('labels', 'data', '*'))

    all_images_list = []
    all_labels_list = []
    all_actions_list = []

    all_x_trains = []
    all_y_trains = []
    all_x_vals = []
    all_y_vals = []
    all_x_tests = []
    all_y_tests = []

    # For each task, find all labeled datasets
    for label_task in label_tasks:
        print(f'Compiling {label_task} files...')
        dataset_addrs = glob.glob(os.path.join(label_task, '*'))

        # Delete previous labels
        os.system(f"rm -rf {os.path.join(label_task, 'images.npy')}")
        os.system(f"rm -rf {os.path.join(label_task, 'labels.npy')}")
        os.system(f"rm -rf {os.path.join(label_task, 'actions.npy')}")

        # For each dataset, compile all images and labels
        for dataset_addr in dataset_addrs:
            # print(f'Compiling images and labels from {dataset_addr}')
            dataset_files = sorted(glob.glob(os.path.join(dataset_addr, '*')))

            # Load all demonstrated actions (not compiled files)
            if dataset_addr[-3:] != 'npy':
                actions = load_actions(
                    labels_dataset_addr=dataset_addr,
                    relative_label_addr=relative_label_addr)

            # Loop for all files in the labeled dataset folder
            for dataset_file in dataset_files:
                file_extension = dataset_file[-4:]

                # Use only files with json labels
                if file_extension == 'json':
                    file_number = dataset_file[-12:-5]

                    # Load labels in numpy format
                    with open(dataset_file) as f:
                        label_json = json.load(f)
                    label_np = np.array(list(label_json.values())[1:])

                    # check if all labels are zeros before appendding (skip images with no labels)
                    if label_np.sum() != 0:
                        all_labels_list.append(label_np)

                        # Only use the actions for the frames we have label
                        all_actions_list.append(actions[int(file_number)])

                        # Load image in numpy format
                        img_addr = dataset_addr + '/' + file_number + '.png'
                        img_np = plt.imread(img_addr)
                        all_images_list.append(img_np)

        # Save all images and all labels to disk
        all_images_np = np.array(all_images_list)
        with open(f'{label_task}/images.npy', 'wb') as f:
            np.save(f, all_images_np)

        all_labels_np = np.array(all_labels_list)
        with open(f'{label_task}/labels.npy', 'wb') as f:
            np.save(f, all_labels_np)

        all_actions_np = np.array(all_actions_list)
        with open(f'{label_task}/actions.npy', 'wb') as f:
            np.save(f, all_actions_np)

        # x_train, y_train, x_val, y_val, x_test, y_test = split_subtask_data(images, labels)

        print_summary(all_labels_np)
        print(f'Done. Saves images.npy and labels.npy in {label_task} folder.')


if __name__ == "__main__":
    main()
