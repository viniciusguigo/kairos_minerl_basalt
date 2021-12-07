import argparse
import glob
import json
import math
import sys

from tqdm import tqdm
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
import gym
import minerl
import os
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler
from skmultilearn.model_selection import iterative_train_test_split


class StateMachineClassifier(nn.Module):
    "Classifier for the MineRL state machine"

    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.n_input_channels = input_shape[0]
        self.num_classes = num_classes

        # features
        self.features = nn.Sequential(
            nn.Conv2d(self.n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )  # output size is 3x16x16 = 768

        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.features(x)
        x = th.flatten(x, 1)
        x = self.classifier(x)
        return x


cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)


# compute accuracy of a trained model given a dataset
def compute_accuracy(model, dataset, loss_function, device, is_val):
    similarity = 0.0
    running_loss = 0.0
    model.eval()
    title = "Val" if is_val else "Test"
    with th.no_grad():
        bar = tqdm(total=len(dataset), file=sys.stdout)
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            pred_labels = model(inputs.to(device))
            cos_result = cosine_similarity(labels.to(device), pred_labels)
            similarity += th.sum(cos_result)
            loss = loss_function(pred_labels, labels.to(device))
            running_loss += loss.item()
            bar.set_description("[{}] loss: {:10.3f} | similarity: {:10.3f}".format(title, loss, th.mean(cos_result)))
            bar.update()
        accuracy = (similarity * 1.0 / len(dataset.dataset))
        bar.set_description("[{}] loss: {:10.3f} | similarity: {:10.3f}".format(title, running_loss / len(dataset), accuracy))
        bar.close()

    accuracy = (similarity * 1.0 / len(dataset.dataset))
    loss = running_loss * 1.0 / len(dataset)
    return loss, accuracy


def load_training_data(all_images_np, all_labels_np, class_threshold):
    column_sums = np.sum(all_labels_np, axis=0)
    print("Label occurrences: {}".format(list(column_sums)))
    columns_less_than_threshold = np.argwhere(column_sums < class_threshold).flatten()
    columns_greater_than_threshold = np.argwhere(column_sums >= class_threshold).flatten()

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

    if columns_less_than_threshold.shape[0] > 0:
        print("Labels missing: {}".format(list(np.array(data_label_names)[columns_less_than_threshold.astype(int)])))
    if columns_greater_than_threshold.shape[0] == 0:
        raise ValueError("Labels are all 0. Check data")
    # print("Labels contained: {}".format(data_label_names[columns_not_all_zero]))

    print("Labels included in dataset")
    label_indices = [(key, val) for key, val in enumerate(data_label_names) if
                     key in set(columns_greater_than_threshold)]
    for classifier_index, (global_index, label_name) in enumerate(label_indices):
        print("Name: {}, Classifier Index: {}, Global Index: {}".format(label_name, classifier_index, global_index))

    data_labels = np.delete(all_labels_np, columns_less_than_threshold, axis=1)
    data_labels = data_labels.astype(np.float32)
    num_classes = columns_greater_than_threshold.shape[0]
    data_images = all_images_np.transpose(0, 3, 1, 2)
    print("Number of classes: {}".format(data_labels.shape[1]))

    return data_images, data_labels, num_classes


def load_training_data_old(data_paths, class_threshold):
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

    column_sums = np.sum(data_labels, axis=0)
    print("Label occurrences: {}".format(list(column_sums)))
    columns_less_than_threshold = np.argwhere(column_sums < class_threshold).flatten()
    columns_greater_than_threshold = np.argwhere(column_sums >= class_threshold).flatten()

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

    if columns_less_than_threshold.shape[0] > 0:
        print("Labels missing: {}".format(list(np.array(data_label_names)[columns_less_than_threshold.astype(int)])))
    if columns_greater_than_threshold.shape[0] == 0:
        raise ValueError("Labels are all 0. Check data")
    # print("Labels contained: {}".format(data_label_names[columns_not_all_zero]))

    print("Labels included in dataset")
    label_indices = [(key, val) for key, val in enumerate(data_label_names) if key in set(columns_greater_than_threshold)]
    for classifier_index, (global_index, label_name) in enumerate(label_indices):
        print("Name: {}, Classifier Index: {}, Global Index: {}".format(label_name, classifier_index, global_index))

    data_labels = np.delete(data_labels, columns_less_than_threshold, axis=1)
    data_labels = data_labels.astype(np.float32)
    num_classes = columns_greater_than_threshold.shape[0]
    data_images = data_images.transpose(0, 3, 1, 2)
    print("Number of classes: {}".format(data_labels.shape[1]))

    return data_images, data_labels, num_classes


def split_data(data_images, data_labels, train_size=0.75, val_size=0.15, test_size=0.1):
    if train_size + val_size + test_size != 1.0:
        raise ValueError("Percentages must sum to 1 ({} + {} + {} != 1.0)".format(train_size, val_size, test_size))
    # split into training and validation
    # x_train, x_val, y_train, y_val = train_test_split(data_images, data_labels, test_size=val_size, random_state=2, stratify=data_labels)
    x_train, y_train, x_val, y_val = iterative_train_test_split(data_images, data_labels, test_size=val_size)

    test_size = 1.0 - train_size / (train_size + test_size)  # 1 - 0.75 / (0.75+0.1)

    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=1, stratify=y_train)
    x_train, y_train, x_test, y_test = iterative_train_test_split(x_train, y_train, test_size=test_size)

    print("data_images: {}".format(data_images.shape))
    print("data_labels: {}".format(data_labels.shape))
    print("x_train: {}".format(x_train.shape))
    print("y_train: {}".format(y_train.shape))
    print("x_test: {}".format(x_test.shape))
    print("y_test: {}".format(y_test.shape))
    print("x_val: {}".format(x_val.shape))
    print("y_val: {}".format(y_val.shape))

    return x_train, y_train, x_val, y_val, x_test, y_test


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
    """Prints summary of labeled data."""
    num_labels = all_labels_np.shape[0]
    print(f'Images labeled: {num_labels} images')
    for i in range(all_labels_np.shape[1]):
        labels_per_class = all_labels_np[all_labels_np[:,i] == 1]
        num_labels_per_class = labels_per_class.shape[0]
        print(f'  Labels for class {i}: {num_labels_per_class} ({100*num_labels_per_class/num_labels:.3f} %)')


def compile_labels():
    # untar labels
    os.system('tar -xvzf labels.tar.gz')
    relative_label_addr = 'labels'

    # Find all labelled tasks
    label_tasks = glob.glob(os.path.join('labels', 'data', '*'))

    all_images_list = []
    all_labels_list = []
    all_actions_list = []

    # For each task, find all labeled datasets
    for label_task in label_tasks:
        print(f'Compiling {label_task} files...')
        dataset_addrs = glob.glob(os.path.join(label_task, '*'))

        # Delete previous labels
        # os.system(f"rm -rf {os.path.join(label_task, 'images.npy')}")
        # os.system(f"rm -rf {os.path.join(label_task, 'labels.npy')}")
        # os.system(f"rm -rf {os.path.join(label_task, 'actions.npy')}")

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
                        # dataset_addr[7:] removes "labels/" from address to get data
                        # directly from the original dataset
                        img_addr = dataset_addr[7:] + '/' + file_number + '.png'
                        img_np = plt.imread(img_addr)
                        all_images_list.append(img_np)

        print(f'Loaded {label_task} data.')

    all_images_np = np.array(all_images_list)
    all_labels_np = np.array(all_labels_list)
    all_actions_np = np.array(all_actions_list)

    print_summary(all_labels_np)
    return all_images_np, all_labels_np, all_actions_np


def save_model(model, model_type, loss, accuracy, epoch):
    if not os.path.exists("train/state_classifier/{}".format(model_type)):
        os.makedirs("train/state_classifier/{}".format(model_type))
    th.save(model, "train/state_classifier/{}/{}_classifier_v1_e{}_l{:.3f}_a{:.3f}.pth".format(model_type, model_type, epoch , loss, accuracy))
    th.save(model.state_dict(), "train/state_classifier/{}/{}_classifier_v1_e{}_l{:.3f}_a{:.3f}_dict.pth".format(model_type, model_type, epoch, loss, accuracy))


def save_best_model(state_classifier: StateMachineClassifier, model_type, testloader, loss_function, device):
    all_model_paths = os.listdir("train/state_classifier/{}".format(model_type))
    all_model_paths = [fname for fname in all_model_paths if "dict" not in fname and ".pth" in fname and "final" not in fname]

    all_model_paths_split = [(fname, float(fname.split("_")[4][1:]), index) for index, fname in enumerate(all_model_paths)]
    best_model_path = sorted(all_model_paths_split, key=lambda x: x[1], reverse=False)[0]
    best_model_state_dict_path = os.path.join("train/state_classifier/{}".format(model_type), best_model_path[0].replace(".pth", "_dict.pth"))

    epoch = int(best_model_path[0].split("_")[3][1:])
    val_loss = float(best_model_path[0].split("_")[4][1:])
    accuracy = float(best_model_path[0].split("_")[5][1:-4])

    state_classifier.load_state_dict(th.load(best_model_state_dict_path))
    state_classifier.to(device)
    state_classifier.eval()

    test_loss, test_accuracy = compute_accuracy(state_classifier, testloader, loss_function, device, False)
    print("Test loss: {:10.3f} | Test similarity: {:10.3f}".format(test_loss, test_accuracy))

    metadata = {
        "epoch": epoch,
        "val_loss": float(val_loss),
        "val_accuracy": float(accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with open("train/state_classifier/best_model_metadata.json", 'w') as jsonfile:
        json.dump(metadata, jsonfile)

    th.save(state_classifier, "train/state_classifier/best_state_classifier.pth")
    th.save(state_classifier.state_dict(), "train/state_classifier/best_state_classifier_dict.pth")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", "-e", type=int, default=50)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3)
    parser.add_argument("--train_size", "-tns", type=float, default=0.8)
    parser.add_argument("--val_size", "-vls", type=float, default=0.1)
    parser.add_argument("--test_size", "-tts", type=float, default=0.1)
    parser.add_argument("--val_interval", "-v", type=int, default=1)
    parser.add_argument("--class_threshold", "-c", type=int, default=50, help="Number of instances of class that must occur to be included in dataset")
    # parser.add_argument("--model_type", "-m", type=str, default="all", choices=["all", "find_cave", "animal_pen", "make_waterfall", "village_house"])

    args = parser.parse_args()
    return args


def train(num_epochs=50, batch_size=32, learning_rate=1e-3, val_interval=1, class_threshold=50, train_size=0.8, val_size=0.1, test_size=0.1):
    all_images_np, all_labels_np, all_actions_np = compile_labels()
    data_images, data_labels, num_classes = load_training_data(all_images_np, all_labels_np, class_threshold=class_threshold)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(data_images, data_labels, train_size=train_size, val_size=val_size, test_size=test_size)

    input_shape = (3, 64, 64)

    # get device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # initialize autoencoder model (and place it on the gpu)
    state_classifier = StateMachineClassifier(input_shape, num_classes).to(device)
    # ae_network = FC_Autoencoder(input_shape).to(device)
    summary(state_classifier, input_shape)

    # setup optimizer and loss
    optimizer = th.optim.Adam(state_classifier.parameters(), lr=learning_rate)
    loss_function = nn.modules.BCELoss()

    # add weighted sampling to help with class imbalance
    class_sample_counts = [len(np.where(y_train[..., i] == 1)[0]) for i in range(num_classes)]
    num_samples = sum(class_sample_counts)
    class_sample_probabilities = [1 - (class_sample_counts[i] / num_samples) for i in range(num_classes)]
    weights = th.tensor(class_sample_probabilities)
    sample_weights = y_train.dot(weights)

    # create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True)

    # convert into pytorch dataset and wrap in a dataloader
    x_train = th.Tensor(x_train)
    x_val = th.Tensor(x_val)
    y_train = th.Tensor(y_train)
    y_val = th.Tensor(y_val)
    x_test = th.Tensor(x_test)
    y_test = th.Tensor(y_test)

    # convert to dataset
    trainset = TensorDataset(x_train, y_train)
    valset = TensorDataset(x_val, y_val)
    testset = TensorDataset(x_test, y_test)

    # wrap in dataloader
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    valloader = DataLoader(valset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    if not os.path.exists("train/state_classifier/all"):
        os.makedirs("train/state_classifier/all")
    with open("train/state_classifier/all/metadata.json", 'w') as jsonfile:
        json.dump({
            "command": ' '.join(sys.argv),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_type": "all",
            "val_interval": val_interval,
            "class_threshold": class_threshold,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,

            "train_data_size": len(x_train),
            "val_data_size": len(x_val),
            "test_data_size": len(x_test),
        }, jsonfile)

    # place network in train mode
    state_classifier.train()

    # train model
    print('Training model')

    minimum_val_loss = sys.maxsize
    for epoch in range(num_epochs):  # loop over dataset num_epoch times
        running_loss = 0.0
        similarity = 0.0

        bar = tqdm(total=math.ceil(len(trainset)/trainloader.batch_size), file=sys.stdout, desc="[Epoch: {}] loss: {:10.3f} | similarity: {:10.3f}".format(epoch + 1, running_loss, similarity))
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = state_classifier(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # compute running training accuracy
            # pred_labels = th.argmax(outputs, dim=1)
            pred_labels = outputs
            # if i == 0:
            #     import IPython
            #     IPython.embed()

            cos_result = cosine_similarity(labels.to(device), pred_labels)
            similarity += th.sum(cos_result)

            # print statistics
            running_loss += loss.item()
            bar.set_description("[Epoch: {}] loss: {:10.3f} | similarity: {:10.3f}".format(epoch + 1, loss, th.mean(cos_result)))
            bar.update()

        # Compute training metrics at the end of each epoch
        acc = (similarity * 1.0 / len(trainset))
        bar.set_description("[Epoch: {}] loss: {:10.3f} | similarity: {:10.3f}".format(epoch + 1, running_loss / len(trainloader), acc))
        bar.close()

        if epoch % val_interval == 0:
            print("\nComputing validation")
            val_loss, val_accuracy = compute_accuracy(state_classifier, valloader, loss_function, device, True)
            if val_loss < minimum_val_loss:
                print("\nValidation loss lower: {:10.3f} < {:10.3f}, saving...".format(val_loss, minimum_val_loss))
                save_model(state_classifier, "all", val_loss, val_accuracy, epoch)
                minimum_val_loss = val_loss
            else:
                print("\nValidation loss higher: {:10.3f} > {:10.3f}".format(val_loss, minimum_val_loss))

    print()
    print('Finished Training')

    # compute training and testing accuracies (this may cause a out of memory error depending on your gpu size)
    # test_acc = compute_accuracy(state_classifier, testset, device)
    # print('Test Accruacy: %.3f', test_acc)
    test_loss, test_accuracy = compute_accuracy(state_classifier, testloader, loss_function, device, False)
    print("Test loss: {:10.3f} | Test similarity: {:10.3f}".format(test_loss, test_accuracy))
    save_model(state_classifier, "all", test_loss, test_accuracy, "final")
    save_best_model(state_classifier, "all", testloader, loss_function, device)


def main():
    print("State Classifier")

    # TODO: load pre-trained autoencoder weights
    # TODO: add early stopping using validation accuracy

    # learning parameters
    args = parse_args()
    num_epochs = args.num_epochs  # 5
    batch_size = args.batch_size  # 32
    learning_rate = args.learning_rate  # 1e-3
    # num_classes = args.num_classes  # 3
    model_type = "all"
    val_interval = args.val_interval
    class_threshold = args.class_threshold
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size

    train(num_epochs=num_epochs,
          batch_size=batch_size,
          learning_rate=learning_rate,
          val_interval=val_interval,
          class_threshold=class_threshold,
          train_size=train_size,
          val_size=val_size,
          test_size=test_size)


if __name__ == "__main__":
    main()
