"""
-------------------------------------------------------------------------------
 Title        : error_distance_right.py
 Description  : Calculate the pointing error for a user and save it in an excel file
 Instructions : Load an image, and then calculate the error between fingertips
 Author       : Guillem Cornella
 Date Created : February 2024
-------------------------------------------------------------------------------
"""

# Import libraries
import cv2
import numpy as np
import mediapipe as mp
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

# Import mediapipe utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def mediapipe_detection(imag, model):
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    im = imag
    im.flags.writeable = False
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    result = model.process(im)
    # Draw the hand annotations on the image.
    im.flags.writeable = True
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im, result

def draw_landmarks(im, result):
    mp_drawing.draw_landmarks(
        im,
        result,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

def calculate_hand_size(left_hand, w, h):
    # To normalize the final outcome distance, calculate the distance between some hand landmark
    # using the Euclidean distance in pixels according to the camera resolution wx720
    if landmark_joints == 'fingertips':
        s = [8, 12, 16, 20, 4, 5, 17]  # starting landmark indices
        e = [5, 9, 13, 17, 2, 0, 0]  # ending landmark indices
    elif landmark_joints == 'MCP':
        s = [5, 17, 17, 2]  # starting landmark indices change for MCP
        e = [0, 0, 5, 0]  # ending landmark indices change for MCP
    elif landmark_joints == 'PIP':
        s = [6, 10, 14, 18, 2, 5, 17]  # starting landmark indices change for PIP
        e = [5, 9, 13, 17, 0, 0, 0]  # ending landmark indices change for PIP
    else:
        print('Choose an appropriate landmark_joints variable name.')
        sys.exit("Strings are not equal, exiting.")
    hand_size = 0
    for i in range(0, len(s)):
        # Calculate the distance in pixels for every line between the selected indeces
        line = np.sqrt(((left_hand[s[i]][0] - left_hand[e[i]][0]) * w) ** 2
                + (h * (left_hand[s[i]][1] - left_hand[e[i]][1])) ** 2)
        hand_size += line

    return hand_size

def calculate_hand_size_mm(xs, ys, zs):
    # To normalize the final outcome distance, calculate the distance between some hand landmark
    # using the Euclidean distance in pixels according to the camera resolution wx720
    s = [8, 12, 16, 20, 4, 5, 17]   # starting indeces
    e = [5, 9, 13, 17, 2, 0, 0]     # ending indeces
    hand_size = 0
    for i in range(0, len(s)):
        # print('distance between: ', s[i], ' and ', e[i])
        # Calculate the distance in pixels for every line between the selected indices (considering xyz)
        line = np.sqrt((xs[s[i]] - xs[e[i]]) ** 2 + (ys[s[i]] - ys[e[i]]) ** 2 + (zs[s[i]] - zs[e[i]]) ** 2)
        # Calculate the distance in pixels for every line between the selected indices (considering xy)
        # line = np.sqrt((xs[s[i]] - xs[e[i]]) ** 2 + (ys[s[i]] - ys[e[i]]) ** 2)
        hand_size += line
    return hand_size


def main():
    global landmark_joints

    user_id = 'test_user'
    landmark_joints = 'fingertips'

    trials_path = '../data/' + user_id + '/'
    file_path = trials_path + 'pr_error_right_' + user_id + '_results.xlsx'

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        # Define column names
        columns = ['user_id', 'assessment', 'target_finger', 'trial', 'coordinates_left', 'coordinates_right',
                   'hand_size_px', 'hand_size_cm', 'hand_size_world_mm', 'distance_px', 'distance_cm']
        # Create an empty DataFrame with predefined column names
        df = pd.DataFrame(columns=columns)

    # Load saved sample data
    f = open(trials_path + 'sample.json')
    data_fing = json.load(f)
    f.close()

    for trial in range(1, int(len(trials_path)/2)-2):
        print('__________________________________________________________________________')
        if (not str(trial) in data_fing) or (
           not os.path.exists(trials_path + str(trial) + '.JPG')):  # if there is no data saved in sample.json
            print('Skip this trial: ', trial)
            continue
        # Load finger target
        print('Trial: ', trial)
        finger_target = data_fing[str(trial)]['right_finger_goal'] # Should be right but it was not changed
        print('finger_target: ', finger_target)

        # for PIP change
        if landmark_joints == 'PIP':
            if finger_target == 5:
                finger_target = 4
            elif finger_target == 9:
                finger_target = 7
            elif finger_target == 13:
                finger_target = 11
            elif finger_target == 17:
                finger_target = 15
            elif finger_target == 21:
                finger_target = 19

        # for MCP change
        elif landmark_joints == 'MCP':
            if finger_target == 5:
                finger_target = 3
            elif finger_target == 9:
                finger_target = 6
            elif finger_target == 13:
                finger_target = 10
            elif finger_target == 17:
                finger_target = 14
            elif finger_target == 21:
                finger_target = 18

        img = cv2.imread(trials_path + str(trial)+'.JPG', cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        # Get image width and height
        width = img.shape[1]
        height = img.shape[0]

        # Resize image for visualization purposes
        # width = int(width/4)
        # height = int(height/4)
        # img = cv2.resize(img, (width, height))

        # Start hand tracking with mediapipe to detect the 2 hands
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.25,
                min_tracking_confidence=0.25,
                max_num_hands=2) as hands:

            # Apply mediapipe HAND detection model
            image, results = mediapipe_detection(img, hands)
            right_exists = False
            if results.multi_hand_landmarks:  # this is a list with [landmarks_hand1, landmarks_hand2]
                R = []  # list that will contain the landmarks from the right hand
                L = []  # list that will contain the landmarks from the left hand
                for rl, hand_landmarks in enumerate(results.multi_hand_landmarks):  # for all the detected hands

                    # Draw the hand landmarks
                    draw_landmarks(image, hand_landmarks)

                    # Get hand index to check label (left or right)
                    handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                    handScore = results.multi_handedness[handIndex].classification[0].score
                    handLabel = results.multi_handedness[handIndex].classification[0].label

                    # No need to flip right and left hand because the saved video is already mirrored
                    # handLabel = 'Right' if handLabel == 'Left' else 'Left'

                    # Initialize a list with the hand label and its score
                    handLandmarks = [[handLabel, handScore]]

                    for landmarks in hand_landmarks.landmark:
                        # Append the handLandmarks list with x, y and z positions of each landmark
                        handLandmarks.append([landmarks.x, landmarks.y, landmarks.z])

                    if handLabel == 'Left':
                        L.append(handLandmarks)
                    else:
                        R.append(handLandmarks)

            if results.multi_hand_world_landmarks:
                Rw = []  # list that will contain the landmarks from the right hand
                Lw = []  # list that will contain the landmarks from the left hand
                xs = []
                ys = []
                zs = []
                for rl, hand_world_landmarks in enumerate(results.multi_hand_world_landmarks):  # for all the detected hands

                    # Get hand index to check label (left or right)
                    handIndex = results.multi_hand_world_landmarks.index(hand_world_landmarks)
                    handScore = results.multi_handedness[handIndex].classification[0].score
                    handLabel = results.multi_handedness[handIndex].classification[0].label
                    print('handLabel: ', handLabel)

                    handWorldLandmarks = []
                    for worldlandmarks in hand_world_landmarks.landmark:
                        #print('Atencio aqui: ')

                        handWorldLandmarks.append([int(worldlandmarks.x*1000), int(worldlandmarks.y*1000)]) #, int(worldlandmarks.z*1000)
                        #print([int(worldlandmarks.x*1000), int(worldlandmarks.y*1000)], ' mm')
                        if handLabel=='Left':
                            Lw.append(handWorldLandmarks)
                        else:
                            right_exists = True
                            Rw.append(handWorldLandmarks)
                            xs.append(int(worldlandmarks.x * 1000))
                            ys.append(int(worldlandmarks.y * 1000))
                            zs.append(int(worldlandmarks.z * 1000))

            if right_exists:
                hand_size_world_mm = calculate_hand_size_mm(xs, ys, zs)
                print('hand_size_world_mm considering xyz: ', hand_size_world_mm)

        # Calculate the coordinates of target and pointing fingers
        if L and R: # if both hands have been detected
            right_index = [int(R[0][finger_target][0] * width), int(R[0][finger_target][1] * height)]
            left_index = [int(L[0][9][0] * width), int(L[0][9][1] * height)]
            print(left_index, right_index)

            # Calculate the euclidean distance (in pixels)
            dx = right_index[0] - left_index[0]
            dy = right_index[1] - left_index[1]
            d_px = int(np.sqrt(dx ** 2 + dy ** 2))
            print('dx: ', dx, ', dy: ', dy)
            print('Distance between fingertips: ', d_px)

            # Get the user's hand size calibration values in cm
            with open('../normalization/' + user_id + '_calib_new_cutTrue.json', 'r') as file:
                data = json.load(file)

            hand_size_cm = data['hand_size_cm']
            print('Right hand size in cm: ', hand_size_cm)

            # Call hand size distance function to get the dimension of the hand in pixels
            hand_size_px = int(calculate_hand_size(R[0][1:], width, height))
            print('Right hand size in pixels: ', hand_size_px)

            d_cm = hand_size_cm/hand_size_px*d_px
            print('Distance in cm is: ', d_cm)

            # Save new trial to dataframe
            # Create a new row to be added
            new_row = {'user_id': user_id, 'assessment':'right_hand',
                       'target_finger' : finger_target, 'trial' : trial,
                       'coordinates_left': left_index, 'coordinates_right':right_index,
                       'dx': dx, 'dy': dy,
                       'hand_size_px':hand_size_px,'hand_size_cm':hand_size_cm, 'hand_size_world_mm':hand_size_world_mm,
                       'distance_px':d_px, 'distance_cm':d_cm}

            # Concatenate the new trial's values to the existing dataframe
            df = pd.concat([df.dropna(axis=1, how='all'),pd.DataFrame([new_row])], ignore_index=True)

            # Show the hand-tracking results
            cv2.imshow(str(trial), image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    df.to_excel(file_path, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()