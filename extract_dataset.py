""" data_processing.py

Recover image data from video dataset and convert all other data to Numpy.

"""
import os
import sys
import glob
import numpy as np
import cv2

from compile_labels import load_actions


# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

all_images_list = []
all_actions_np = None

# Find all MineRLBasalt tasks
tasks = glob.glob(os.path.join(MINERL_DATA_ROOT, 'MineRLBasalt*'))

# For each task, find all dataset
for task in tasks:
    del all_images_list
    del all_actions_np
    all_images_list = []
    all_actions_np = None
    print(f'Extract all data from {task}')
    dataset_addrs = glob.glob(os.path.join(task, '*'))

    # Delete previous data
    os.system(f"rm -rf {os.path.join(task, 'images.npy')}")
    os.system(f"rm -rf {os.path.join(task, 'actions.npy')}")
    
    # Process data of each dataset
    for dataset_addr in dataset_addrs:
        # Load all demonstrated actions (not compiled files)
        if dataset_addr[-3:] != 'npy':
            print(f'  Extracting data from {dataset_addr}')

            try:
                # Load image data
                cap = cv2.VideoCapture(os.path.join(dataset_addr, 'recording.mp4'))
                frame_counter = 0

                # Load action data
                actions = load_actions(
                    labels_dataset_addr=None,
                    relative_label_addr=None,
                    dataset_addr=dataset_addr)
                if all_actions_np is None:
                    # First array of actions
                    all_actions_np = actions
                else:
                    # Not the first, stack with previous
                    all_actions_np = np.vstack((all_actions_np, actions))

                # Read until video is completed
                while(cap.isOpened()):
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if ret == True:                
                        # Store frame
                        all_images_list.append(frame)
                        
                        # Next frame
                        frame_counter += 1

                    # Break the loop
                    else: 
                        break
            except:
                print(f'    Invalid dataset (no video data)')

    # Save all images and all labels to disk
    all_images_np = np.array(all_images_list, dtype = np.float32)
    
    # convert images to 0-1 and fix rbg order
    all_images_np = all_images_np / 255.0
    all_images_np = all_images_np[:,:,:,[2,1,0]]
    
    print(f'    [*] Extracted {all_images_np.shape[0]} images for task {task}')
    with open(f'{task}/images.npy', 'wb') as f:
        np.save(f, all_images_np)
    del all_images_np
    print(f'    [*] Extracted {all_actions_np.shape[0]} actions for task {task}')
    with open(f'{task}/actions.npy', 'wb') as f:
        np.save(f, all_actions_np)