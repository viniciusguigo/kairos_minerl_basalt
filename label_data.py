""" label_data.py

Basic GUI to load expert dataset and label visual and trajectory features.

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
parser.add_argument("--frame_skip", type=int, default=1)
parser.add_argument("--label_all", type=str, default=None)
parser.add_argument("--quick_label", action="store_true", default=False)

# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
BASALT_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLBasaltFindCave-v0')
ROOT_DATASET_ADDR = os.path.join(MINERL_DATA_ROOT, BASALT_GYM_ENV)


class KAIROS_Label_GUI():
    """
    Displays agent POV and internal states relevant when debugging.
    """
    def __init__(self, dataset_addr, labels_addr, use_yolo_labels):
        self.resolution = 512 # pixels
        self.offset = 10
        self.button_distance = 60
        self.top_bar_height = int(self.resolution/6)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.text_color = (255, 255, 255)
        self.indicator_color = (0, 255, 0)
        self.font_thickness = 1
        self.window_name = 'KAIROS MineRL Label GUI'
        self.dataset_addr = dataset_addr
        self.labels_addr = labels_addr
        self.use_yolo_labels = use_yolo_labels

        self.indicator_text_pos_x = 10*self.offset+self.resolution
        self.indicator_pos_x = self.indicator_text_pos_x-4*self.offset
        self.indicator_pos_y = int(self.offset+self.resolution/6)
        self.indicator_radius = 20
        self.indicator_names = [
            '[n] NONE',
            '[c] HAS_CAVE',
            '[i] INSIDE_CAVE',
            '[d] DANGER_AHEAD',
            '[m] HAS_MOUNTAIN',
            '[f] FACING_WALL',
            '[t] AT_THE_TOP',
            '[w] GOOD_WATERFALL_VIEW',
            '[p] GOOD_PEN_VIEW',
            '[h] GOOD_HOUSE_VIEW',
            '[a] HAS_ANIMALS',
            '[o] HAS_OPEN_SPACE',
            '[s] ANIMALS_INSIDE_PEN',
        ]
        self.num_indicators = len(self.indicator_names)
        self.map_key_to_label = {
            ' ': -2, 'q': -1, 'n': 0, 'c': 1, 'i': 2, 'd': 3, 'm': 4,
            'f': 5, 't': 6, 'w': 7, 'p': 8, 'h': 9, 'a': 10,
            'o': 11, 's': 12
        }

        self.font_thickness_box = 1
        self.trackbar_position = 0
        self.num_box_classes = 1
        self.box_classes_map = {
            0: "CAVE",
            1: "WATERFALL",
        }
        self.box_classes_color_map = {
            "CAVE": (86, 105, 204),
            "WATERFALL": (230, 219, 122),
        }
        self._update_box_class_label()
        self.yolo_labels = []

        self.mouse_x_start, self.mouse_y_start = 0, 0
        self.mouse_x_end, self.mouse_y_end = 0, 0
        self.DRAW_BOX = True
        self.frame = None

    def _update_box_class_label(self):
        self.current_box_label = self.box_classes_map[self.trackbar_position]
        self.label_color = (255,255,255)#self.box_classes_color_map[self.current_box_label]
        self.box_color = self.box_classes_color_map[self.current_box_label]

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

        # Save labels in YOLO format
        if self.use_yolo_labels:
            yolo_label_addr = os.path.join(self.labels_addr, f'{frame_counter}.txt')
            with open(yolo_label_addr, 'w') as outfile:
                for yolo_label in self.yolo_labels:
                    outfile.write(yolo_label+"\n")

        # Reset YOLO labels for next frame
        self.yolo_labels = []


    def update_labels(self, button_id):
        if button_id == 0:
            self.labels['none'] = 1
        elif button_id == 1:
            self.labels['has_cave'] = 1
        elif button_id == 2:
            self.labels['inside_cave'] = 1
        elif button_id == 3:
            self.labels['danger_ahead'] = 1
        elif button_id == 4:
            self.labels['has_mountain'] = 1
        elif button_id == 5:
            self.labels['facing_wall'] = 1
        elif button_id == 6:
            self.labels['at_the_top'] = 1
        elif button_id == 7:
            self.labels['good_waterfall_view'] = 1
        elif button_id == 8:
            self.labels['good_pen_view'] = 1
        elif button_id == 9:
            self.labels['good_house_view'] = 1
        elif button_id == 10:
            self.labels['has_animals'] = 1
        elif button_id == 11:
            self.labels['has_open_space'] = 1
        elif button_id == 12:
            self.labels['animals_inside_pen'] = 1

    def augment_frame(self, frame, frame_counter):
        # Keep original frame
        self.original_frame = copy.deepcopy(frame)

        # Reset labels
        # '[n] NONE',
        # '[c] HAS_CAVE',
        # '[i] INSIDE_CAVE',
        # '[d] DANGER_AHEAD',
        # '[m] HAS_MOUNTAIN',
        # '[f] FACING_WALL',
        # '[t] AT_THE_TOP',
        # '[w] GOOD_WATERFALL_VIEW',
        # '[p] GOOD_PEN_VIEW',
        # '[h] GOOD_HOUSE_VIEW',
        # '[a] HAS_ANIMALS',
        # '[o] HAS_OPEN_SPACE',
        # '[s] ANIMALS_INSIDE_PEN',
        self.labels = {
            'image_id': frame_counter,
            'none': 0,
            'has_cave': 0,
            'inside_cave': 0,
            'danger_ahead': 0,
            'has_mountain': 0,
            'facing_wall': 0,
            'at_the_top': 0,
            'good_waterfall_view': 0,
            'good_pen_view': 0,
            'good_house_view': 0,
            'has_animals': 0,
            'has_open_space': 0,
            'animals_inside_pen': 0,
        }

        # Resize for visualization
        frame = cv2.resize(frame, dsize=[self.resolution, self.resolution])

        # Add top and right panel for instructions and labels
        frame = cv2.copyMakeBorder(
            frame,
            top=self.top_bar_height,
            bottom=self.top_bar_height*4,
            left=0,
            right=int(self.resolution/1.5),
            borderType=cv2.BORDER_CONSTANT)

        # Add Header
        dataset_addr_position = (self.offset, 3*self.offset)
        frame = cv2.putText(frame, f'DATASET: {self.dataset_addr}', dataset_addr_position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)
        
        frame_counter_position = (self.offset, 6*self.offset)
        frame = cv2.putText(frame, f'FRAME #{frame_counter}', frame_counter_position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        # Add buttons with labels
        visual_features_title_position = (self.offset+self.resolution, self.offset+int(self.resolution/6))
        frame = cv2.putText(frame, 'VISUAL FEATURES:', visual_features_title_position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        for i in range(self.num_indicators):
            frame = self.create_button(frame, title=self.indicator_names[i], button_id=i+1)

        # Add trackbar to select bounding box classes
        if self.use_yolo_labels:
            frame = self.display_current_box_label(frame=frame)

        # Update internal frame for mouse events
        self.frame = frame

        return self.frame

    def is_inside_button(self):
        # Check each indicator button
        for i in range(self.num_indicators):
            inside_x_range = False
            inside_y_range = False

            # Check x coordinate
            if (self.mouse_x_start > (self.indicator_pos_x-self.indicator_radius)) and \
                (self.mouse_x_start < (self.indicator_pos_x+self.indicator_radius)):
                inside_x_range = True

            # Check y coordinate
            button_y_position = self.indicator_pos_y+(i+1)*self.button_distance
            if (self.mouse_y_start > (button_y_position-self.indicator_radius)) and \
                (self.mouse_y_start < (button_y_position+self.indicator_radius)):
                inside_y_range = True

            # Update indicator color, if clicked inside
            if inside_x_range and inside_y_range:
                selection_button = (self.indicator_pos_x, button_y_position)
                self.frame = cv2.circle(self.frame, selection_button, self.indicator_radius, self.indicator_color, -1)

                # Update displayed frame
                cv2.imshow(self.window_name, self.frame)

                # Update labels
                self.update_labels(button_id=i)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x_start = x
            self.mouse_y_start = y

            # Check if inside any button
            self.is_inside_button()

        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_x_end = x
            self.mouse_y_end = y

            # Check if box is out of bounds
            last_click_inside_image = True
            if self.mouse_x_end > self.resolution or \
            self.mouse_y_end < self.top_bar_height or \
            self.mouse_y_start < self.top_bar_height:
                last_click_inside_image = False

            # Draw rectangle when relesing mouse button
            if self.DRAW_BOX and last_click_inside_image:
                # Draw box
                self.frame = cv2.rectangle(
                    self.frame, 
                    pt1=(self.mouse_x_start, self.mouse_y_start),
                    pt2=(self.mouse_x_end, self.mouse_y_end),
                    color=self.box_color, thickness=2)

                # Add text based on current label
                label_position = (self.mouse_x_start, self.mouse_y_start-5)
                self.frame = cv2.putText(self.frame, self.current_box_label, label_position, self.font, 
                   self.font_scale, self.label_color, self.font_thickness_box, cv2.LINE_AA)
                
                # Update displayed frame
                cv2.imshow(label_gui.window_name, label_gui.frame)

                # Save YOLO labels to be saved before going to the next frame
                # YOLO format: <object-class> <x> <y> <width> <height>
                x = int(np.min([self.mouse_x_start, self.mouse_x_end])*64/self.resolution)
                y = int(np.max([self.mouse_y_start, self.mouse_y_end])*64/self.resolution)
                width = int(np.abs(self.mouse_x_start-self.mouse_x_end)*64/self.resolution)
                height = int(np.abs(self.mouse_y_start-self.mouse_y_end)*64/self.resolution)
                current_yolo_label = f'{self.trackbar_position} {x} {y} {width} {height}'
                self.yolo_labels.append(current_yolo_label)

    def trackbar_callback(self):
        self.trackbar_position = cv2.getTrackbarPos('Select Class:', self.window_name)
        self._update_box_class_label()

        # Update displayed 
        self.frame = self.display_current_box_label(frame=self.frame)
        cv2.imshow(label_gui.window_name, label_gui.frame)

    def display_current_box_label(self, frame):
        # Draw black box to cover previous label
        class_title_position = (self.offset+self.resolution, int(self.resolution+0.75*self.top_bar_height))
        frame = cv2.rectangle(
                    frame, 
                    pt1=(class_title_position[0]-self.offset, int(class_title_position[1]-1.5*self.offset)),
                    pt2=(int(1.5*self.resolution), int(self.resolution+self.top_bar_height)),
                    color=(0, 0 ,0), thickness=-1)

        # Display class selected using the trackbar
        frame = cv2.putText(frame, f'CLASS: {self.current_box_label}', class_title_position, self.font, 
                   self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        return frame


if __name__ == "__main__":
    USE_YOLO_LABELS = False

    args = parser.parse_args()
    relative_dataset_addr = args.relative_dataset_addr

    # Loop for all selected dataset
    if args.label_all is None:
        relative_dataset_addrs = [args.relative_dataset_addr]
    else:
        relative_dataset_addrs = glob.glob(os.path.join(MINERL_DATA_ROOT, args.label_all, '*'))

    for relative_dataset_addr in relative_dataset_addrs:

        # Setup path for labels
        labels_addr = os.path.join('labels', relative_dataset_addr)
        os.makedirs(labels_addr, exist_ok=True)

        print(f'Labeling dataset {relative_dataset_addr} skipping {args.frame_skip} frames')
        
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
        label_gui = KAIROS_Label_GUI(
            dataset_addr=relative_dataset_addr,
            labels_addr=labels_addr,
            use_yolo_labels=USE_YOLO_LABELS) 
        cv2.namedWindow(label_gui.window_name)
        cv2.setMouseCallback(label_gui.window_name, label_gui.mouse_callback)

        # Trackbar for YOLO labels
        if USE_YOLO_LABELS:
            cv2.createTrackbar('Select Class:', label_gui.window_name, 0, label_gui.num_box_classes, lambda x: label_gui.trackbar_callback())

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Check if skipping frames
                if frame_counter % args.frame_skip == 0:
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

                    # Quick label mode enables user to label using keyboard. Limitation: one label per frame.
                    if args.quick_label:
                        button_id = label_gui.map_key_to_label[chr(key_pressed)]
                        label_gui.update_labels(button_id)

                    # Pressed button to go to next image, save labels
                    label_gui.save_labels(frame_counter)
                
                # Next frame
                frame_counter += 1

            # Break the loop
            else: 
                break