""" label_subtasks.py

Basic GUI to load expert dataset and label subtasks for each task.

"""
import argparse
import os, sys
import json
import glob
import copy
import numpy as np
import cv2
import minerl


parser = argparse.ArgumentParser()
parser.add_argument("--relative_dataset_addr", type=str, default=None)
parser.add_argument("--label_all", type=str, default=None)

# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
BASALT_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLBasaltFindCave-v0')
ROOT_DATASET_ADDR = os.path.join(MINERL_DATA_ROOT, BASALT_GYM_ENV)


class KAIROS_Subtask_Label_GUI():
    """
    Displays agent POV and internal states relevant when debugging.
    """
    def __init__(self, dataset_addr, labels_addr):
        self.resolution = 512 # pixels
        self.offset = 10
        self.button_distance = 60
        self.top_bar_height = int(self.resolution/6)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.text_color = (255, 255, 255)
        self.font_thickness = 1
        self.window_name = 'KAIROS MineRL Subtask Label GUI'
        self.dataset_addr = dataset_addr
        self.labels_addr = labels_addr

        self.indicator_text_pos_x = 10*self.offset+self.resolution
        self.indicator_pos_x = self.indicator_text_pos_x-4*self.offset
        self.indicator_pos_y = int(self.offset+self.resolution/6)
        self.indicator_radius = 20
        self.task_name = self.extract_task_name(self.dataset_addr)
        self.indicator_names_map = {
            'FindCave': [
                '0) Find cave',
                '1) Go to cave',
                '2) End episode (Inside cave)',
            ],
            'MakeWaterfall': [
                '0) Find spot to build waterfall',
                '1) Build waterfall',
                '2) Go to picture location',
                '3) End episode (Looking at waterfall)',
            ],
            'CreateVillageAnimalPen': [
                '0) Find animals',
                '1) Find spot to build pen',
                '2) Build pen',
                '3) Lure animals',
                '4) End episode (Looking at pen)',
            ],
            'BuildVillageHouse': [
                '0) Find spot to build house',
                '1) Build house',
                '2) Tour house',
                '3) End episode (Looking at house)',
            ]
        }
        self.task_id_map = {
            'FindCave': 1,
            'MakeWaterfall': 2,
            'CreateVillageAnimalPen': 3,
            'BuildVillageHouse': 4,
        }
        self.task_id = self.task_id_map[self.task_name]
        self.indicator_names = self.indicator_names_map[self.task_name]
        self.num_indicators = len(self.indicator_names)
        self.map_key_to_label = {' ': -2, 'q': -1}

        self.font_thickness_box = 1
        self.trackbar_position = 0
        self.num_subtasks = len(self.indicator_names)-1
        self._update_subtask_label()
        self.frame = None


    def _update_subtask_label(self):
        self.current_subtask_label = self.indicator_names[self.trackbar_position]


    def create_button(self, frame, title, button_id):
        position = (self.indicator_text_pos_x, self.indicator_pos_y+button_id*self.button_distance)
        # Add text
        frame = cv2.putText(frame, title, position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)
        # Add button to click
        selection_button = (self.indicator_pos_x, position[1])
        frame = cv2.circle(frame, selection_button, self.indicator_radius, self.text_color, -1)

        return frame


    def save_labels(self, frame_counter):
        # Pad frame counter to ease sorting
        frame_counter = str(frame_counter).zfill(7)

        # Save frame
        image_addr = os.path.join(self.labels_addr, f'{frame_counter}.png')
        cv2.imwrite(image_addr, self.original_frame)

        # Save labels
        label_addr = os.path.join(self.labels_addr, f'{frame_counter}.json')
        with open(label_addr, 'w') as outfile:
            json.dump(self.labels, outfile)


    def reset_label(self):
        self.labels = {
            'image_id': self.frame_counter,
            'task_id': self.task_id,
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
        }


    def augment_frame(self, frame, frame_counter):
        # Keep original frame
        self.frame_counter = frame_counter
        self.original_frame = copy.deepcopy(frame)

        # Update labels
        self.update_label()

        # Resize for visualization
        frame = cv2.resize(frame, dsize=[self.resolution, self.resolution])

        # Add top and right panel for instructions and labels
        frame = cv2.copyMakeBorder(
            frame,
            top=self.top_bar_height,
            bottom=0,
            left=0,
            right=self.resolution,
            borderType=cv2.BORDER_CONSTANT)

        # Add Header
        dataset_addr_position = (self.offset, 3*self.offset)
        frame = cv2.putText(frame, f'DATASET: {self.dataset_addr}', dataset_addr_position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)
        
        frame_counter_position = (self.offset, 6*self.offset)
        frame = cv2.putText(frame, f'FRAME #{frame_counter}', frame_counter_position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        # Add buttons with labels
        subtask_title_position = (self.offset+self.resolution, self.offset+int(self.resolution/6))
        frame = cv2.putText(frame, 'SUBTASKS:', subtask_title_position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        for i in range(self.num_indicators):
            frame = self.create_button(frame, title=self.indicator_names[i], button_id=i+1)

        # Add trackbar to select bounding box classes
        frame = self.display_current_subtask_label(frame=frame)

        # Update internal frame for mouse events
        self.frame = frame

        return self.frame

    def update_label(self):
        self.reset_label()
        self.labels[str(self.trackbar_position)] = 1


    def trackbar_callback(self):
        self.trackbar_position = cv2.getTrackbarPos('Select Subtask:', self.window_name)
        self._update_subtask_label()
        
        # Update label value
        self.update_label()

        # Update displayed 
        self.frame = self.display_current_subtask_label(frame=self.frame)
        cv2.imshow(label_gui.window_name, label_gui.frame)


    def display_current_subtask_label(self, frame):
        # Draw black box to cover previous label
        class_title_position = (self.offset+self.resolution, int(self.resolution+0.75*self.top_bar_height))
        frame = cv2.rectangle(
                    frame, 
                    pt1=(class_title_position[0]-self.offset, int(class_title_position[1]-1.5*self.offset)),
                    pt2=(int(2*self.resolution), int(self.resolution+self.top_bar_height)),
                    color=(0, 0 ,0), thickness=-1)

        # Display class selected using the trackbar
        frame = cv2.putText(frame, f'CURRENT SUBTASK: {self.current_subtask_label}', class_title_position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        return frame

    def extract_task_name(self, s, first='MineRLBasalt', last='-v0'):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""



if __name__ == "__main__":
    args = parser.parse_args()
    relative_dataset_addr = args.relative_dataset_addr

    # Loop for all selected dataset
    if args.label_all is None:
        relative_dataset_addrs = [args.relative_dataset_addr]
    else:
        relative_dataset_addrs = glob.glob(os.path.join(MINERL_DATA_ROOT, args.label_all, '*'))

    for relative_dataset_addr in relative_dataset_addrs:
        # Setup path for labels
        labels_addr = os.path.join('subtask_labels', relative_dataset_addr)
        os.makedirs(labels_addr, exist_ok=True)

        print(f'Labeling dataset {relative_dataset_addr}')
        
        # Load metadata information
        metadata_file = open(os.path.join(relative_dataset_addr, 'metadata.json'), 'r')
        metadata = json.load(metadata_file)
        metadata_file.close()
        print(f'Metadata: {metadata}')

        # Load low-level observation
        lowlvl_data = np.load(os.path.join(relative_dataset_addr, 'rendered.npz'))

        # Load image data
        cap = cv2.VideoCapture(os.path.join(relative_dataset_addr, 'recording.mp4'))
        frame_counter = 0

        # Initialize labeling GUI
        label_gui = KAIROS_Subtask_Label_GUI(
            dataset_addr=relative_dataset_addr,
            labels_addr=labels_addr) 
        cv2.namedWindow(label_gui.window_name)

        # Trackbar for subtask labels
        cv2.createTrackbar(
            'Select Subtask:', label_gui.window_name, 0, label_gui.num_subtasks,
            lambda x: label_gui.trackbar_callback())

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Augment frame using GUI features
                frame = label_gui.augment_frame(frame, frame_counter)           

                # Display the resulting frame
                cv2.imshow(label_gui.window_name, label_gui.frame)

                # Only advances frame after a valid key press
                key_pressed_valid = False
                while not key_pressed_valid:
                    key_pressed = cv2.waitKey(0)
                    if chr(key_pressed) in label_gui.map_key_to_label.keys():
                        key_pressed_valid = True

                # Press Q on keyboard to exit
                if key_pressed & 0xFF == ord('q'):
                    break

                # Pressed button to go to next image, save labels
                label_gui.save_labels(frame_counter)
                
                # Next frame
                frame_counter += 1

            # Break the loop
            else: 
                break