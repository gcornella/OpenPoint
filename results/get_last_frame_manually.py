"""
-------------------------------------------------------------------------------
 Title        : get_last_frame_manually.py
 Description  : Extract a frame manually from a video and save it as an image
 Instructions : Press e - next trial
                Press s - next frame
                Press a - previous frame
                Press w - save frame to an image and go to the next trial
                Press q - destroy windows and exit
 Author       : Guillem Cornella
 Date Created : February 2024
-------------------------------------------------------------------------------
"""

import cv2
import os

width, height = (1280, 720)

user_id = 'test_user'
trials_path = '../data/' + user_id + '/'


# Create folder to save the last frames where they guessed the position of the fingers
if not os.path.exists(trials_path+'last_frames'):
    # If the folder does not exist, create it
    os.makedirs(trials_path+'last_frames')
    print("Folder created")

for trial in range(1, 30):
    print('Trial: ', trial)
    if not os.path.exists(trials_path + str(trial) + '.AVI'):
        print('Skip this trial: ', trial)
        continue
    video_path = os.path.join(trials_path, str(trial))+'.avi'

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    frame = 1
    print('Total number of frames: ',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    success, image = cap.read()
    while cap.isOpened():
        if not success:
            print("Video finished.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Show the video image with all the variations
        cv2.imshow("Image_UP", image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            print('s pressed - FORWARD')
            frame += 1
            success, image = cap.read()
            continue

        elif key == ord('a'):  # for different keyboard layouts
            # print('a pressed - BACK')
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 2)
            frame -= 1
            success, image = cap.read()

        # save data from this position/frame
        elif key == ord('w'):  # for different keyboard layouts
            print('w pressed - SAVING')
            cv2.imwrite(trials_path+'last_frames/'+ str(trial)+'.JPG', image)
            break

        elif key == ord('e'):
            print('e pressed - next pic')
            break

        elif key == ord('q'):
            cv2.destroyWindow('Image_UP')

        else:
            print('This key does nothing')

