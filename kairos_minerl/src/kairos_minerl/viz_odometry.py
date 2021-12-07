import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import time

EXP_ID = '2021-09-10 16:48:03.161207'

# Load odometry data
odom_addr = f'train/odometry/odometry_log_{EXP_ID}.csv'
odom_data_df = pd.read_csv(odom_addr, delimiter=',')

# Setup odometry as a image
map_res = 10
min_x = np.abs(odom_data_df['x'].min())
min_y = np.abs(odom_data_df['y'].min())

# Load video data
video_addr = f'train/videos/kairos_minerl_{EXP_ID}.mp4'
cap = cv2.VideoCapture(video_addr)
frame_counter = 1

# Setup video recorder
out = cv2.VideoWriter(f'tmp/odom_{EXP_ID}.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1024, 512))

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:                
        # Display odometry as a frame
        # Convert odometry to pixels
        x = (map_res*(odom_data_df['x'].iloc[:frame_counter]+min_x)).astype(int).values
        y = (map_res*(odom_data_df['y'].iloc[:frame_counter]+min_y)).astype(int).values
        
        # Setup odometry image with maximum x or y dimension
        max_coord = max(x.max(), y.max())
        odom_img = np.zeros((max_coord+1, max_coord+1, 3), np.uint8)

        # Substitute coordinates as white pixels
        odom_img[x, y] = 255

        # Add circle to current robot position
        x_pos = y[-1]
        y_pos = x[-1]
        odom_img = cv2.circle(odom_img, (x_pos, y_pos), 5, (0,255,0), -1)

        # Make sure image always has the same size
        odom_img = cv2.resize(odom_img, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Add text with odometry info
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        font_scale = 0.5
        text_color = (255, 255, 255)

        odom_curr_x = odom_data_df['x'].iloc[frame_counter]
        odom_curr_y = odom_data_df['y'].iloc[frame_counter]
        odom_curr_heading = odom_data_df['heading'].iloc[frame_counter]

        odom_img = cv2.putText(odom_img, f'x: {odom_curr_x:.2f} m', (10, 20), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        odom_img = cv2.putText(odom_img, f'y: {odom_curr_y:.2f} m', (10, 40), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)
        odom_img = cv2.putText(odom_img, f'heading: {odom_curr_heading:.2f} deg', (10, 60), font, 
                   font_scale, text_color, thickness, cv2.LINE_AA)

        # Concatenate with frame and display
        frame_display = np.concatenate((frame, odom_img), axis=1)

        # Write to disk
        out.write(frame_display)

        # # Display image
        # cv2.imshow('frame', frame_display)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        
        # Next frame
        frame_counter += 1

    # Break the loop
    else: 
        break

# Release video capture
out.release()