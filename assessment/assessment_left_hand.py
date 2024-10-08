"""
-------------------------------------------------------------------------------
 Title        : assessment_left_hand.py
 Description  : This can be used as an exercise to assess hand proprioception
 Instructions : 1. Place the left hand on your chest
                2. Try to touch each finger with the index fingertip from the right hand
 Author       : Guillem Cornella
 Date Created : February 2024
-------------------------------------------------------------------------------
"""

# Import libraries
import cv2
import time
import numpy as np
import random
import statistics
import itertools
import json
from threading import Thread
import mediapipe as mp
import tkinter as tk
import datetime
import os

# Import MediaPipe utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation


# This background task performs the countdown for the start green button
def countdown_task():
    global activate_countdown, time_beginning_trial, show_green, countdown, trial
    countdown = 3
    while True:
        time.sleep(0.1)  # To avoid the program to freeze
        if activate_countdown:
            # Set a countdown of 3 seconds
            for countdown in [2, 1, 0]:
                # A trial starts when the countdown gets a value of 0
                if countdown == 0:
                    trial += 1
                    print('----------- Start trial {} ----------------'.format(trial))
                    time_beginning_trial = time.perf_counter()
                    activate_countdown = False
                    show_green = False
                    time.sleep(3)
                    break
                if activate_countdown is False:
                    break
                time.sleep(0.75)


# This background tasks checks if seven seconds have elapsed form the start of the trial.
# Each trial has a maximum duration of 7 seconds, but it can stop earlier if detecting steady error distance 
def trial_time_limit_task():
    global countdown, sevenSecondStop, endTrial
    while True:
        time.sleep(0.1)  # To avoid the program to freeze
        if countdown == 0:
            start_time = time.time()
            # Create an infinite for loop that can be broken. Break breaks both while loops.
            sequence = [0, 1]
            for item in itertools.cycle(sequence):
                current = time.time()
                time.sleep(0.01)
                if current - start_time >= 7:
                    sevenSecondStop = True
                    time.sleep(2)
                    countdown = 3
                    sevenSecondStop = False
                    break
                if endTrial:
                    break


# This class uses tkinter to display an input box where the user can define the name and other settings.
class Input:
    def __init__(self):
        self.now = datetime.datetime.now()  # Get the date

        # Top level window
        self.frame = tk.Tk()
        self.frame.title("User ID Input")
        self.frame.geometry('250x150')

        # DateBox Creation
        self.dateLabel = tk.Label(self.frame, text="Date: ")
        self.dateLabel.grid(row=0, column=0)
        self.dateText = tk.Label(self.frame, text=self.now.strftime("%Y-%m-%d"))
        self.dateText.grid(row=0, column=1)

        # NameBox creation
        self.nameLabel = tk.Label(self.frame, text="Name: ")
        self.nameLabel.grid(row=1, column=0)
        self.nameText = tk.Text(self.frame, height=3, width=20)
        self.nameText.grid(row=1, column=1)

        # Button Creation
        self.printButton = tk.Button(self.frame, text="Start", command=self.print_input)
        self.printButton.grid(row=3, column=1)

        # Result Label Creation
        self.lbl = tk.Label(self.frame, text="")
        self.lbl.grid(row=4, column=1)

        self.frame.mainloop()
        self.experiment = {}

    # Function for getting Input from textbox and printing it at label widget
    def print_input(self):
        global inp
        inp = self.nameText.get(1.0, "end-1c")
        self.lbl.config(text="Provided Input: " + inp)
        # Close the window called frame
        self.frame.destroy()

    @staticmethod
    def create_user_folder(name):
        filepath = '../data/' + name
        # if the user does not exist, create a new directory to store the data
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        return

    def run(self):
        global inp
        self.experiment['date'] = self.now.strftime("%Y-%m-%d")
        self.experiment['name'] = inp
        self.create_user_folder(inp)
        return self.experiment


# Main class where the program is executed
class App:
    def __init__(self):
        self.display_countdown = None
        self.time_start_execution = time.perf_counter()
        self.time_every_thirty_seconds = time.perf_counter()
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.original_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.finger_coordinates_list = [(1117, 25), (1234, 79), (1258, 132), (1248, 175), (1207, 224)]
        self.finger_coordinates = random.choice(self.finger_coordinates_list)

        self.l_idx = []
        self.l_idx_list = []
        self.r_idx_8 = []
        self.LR = []  # list that will contain both landmarks from left and right hands
        self.total_dict = {'date': experiment['date'], 'width': width, 'height': height}

        # Variables used to calculate error distance related issues
        self.std = 100.0    # Initialize the std of the my_array to calculate when finger distance is steady
        self.dist_xy = 0    # distance variable between xy coordinates of the index fingertip

        # Rolling array method to save new value in the next available position, keeping old values in memory
        self.current_idx = 0
        # Initialize array with random values from 0 to 1000
        self.max_size = 50
        self.my_array = [int(1000 * random.random()) for i in range(self.max_size)]

        # Lists containing info from the entire trial
        self.all_distances = []
        self.all_times = []
        self.all_coord = []

        self.results_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.seven_second_warning = cv2.putText(self.results_image, 'Next trial', (200, 300),
                                                cv2.FONT_HERSHEY_SIMPLEX, 5,
                                                (255, 255, 255), 2, cv2.LINE_AA)

        # Variables used for masking the right hand
        self.structuring_element = (80, 80)  # Choose structuring elements
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.structuring_element)  # Define the kernel

    # Define a function for the mediapipe detection
    @staticmethod
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

    # Define a function to draw the landmarks on top of the hands
    @staticmethod
    def draw_landmarks(im, result):
        mp_drawing.draw_landmarks(
            im,
            result,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Define a function to hide the left hand from the user's view
    def hide(self, imag, contour):
        # Calculate the convex hull of the contour
        hull = cv2.convexHull(np.array(contour))
        # Create a single channel black image
        mask = np.zeros((height, width))
        # Create a mask using the convex hull
        cv2.fillPoly(mask, pts=[hull], color=(255, 255, 255))
        mask = mask.astype('uint8')
        # Increase the mask using dilation to blur the contours of the hand
        mask = cv2.dilate(mask, self.kernel, iterations=1)  # Dilate with the structuring element chosen
        # Instead of applying inpainting to blur (which is slow), convert the pixels to black
        resulting = cv2.bitwise_and(imag, imag, mask=cv2.bitwise_not(mask))

        # To draw the landmarks on top of the hand, uncomment the following lines of code
        # resulting = cv2.flip(resulting, 1)
        # draw_landmarks(resulting, results)
        # resulting = cv2.flip(resulting, 1)
        return resulting

    # This function saves the experiment's info into a specified directory
    def save_data(self):
        global trial, finger_number, experiment, right_finger_indeces
        print('Saving data...')

        save_dict = {"trial": trial, "distances": self.all_distances, "times": self.all_times,
                     "all_coord": self.all_coord, "left_finger_goal": right_finger_indeces[finger_number]}
        if endTrial:
            save_dict["finished"] = 'finished'
        if sevenSecondStop:
            save_dict["finished"] = 'not finished'

        # Save the coordinates of both hand landmarks at the final frame of the trial
        if len(self.LR) == 2:
            if self.LR[0][0][0] == 'Left':
                save_dict["LeftHand"] = self.LR[0]
                save_dict["RightHand"] = self.LR[1]
            else:
                save_dict["RightHand"] = self.LR[0]
                save_dict["LeftHand"] = self.LR[1]
        else:
            if self.LR[0][0] == 'Left':
                save_dict["LeftHand"] = self.LR
                save_dict["RightHand"] = []
            else:
                save_dict["RightHand"] = self.LR
                save_dict["LeftHand"] = []

        # Append as save all trial data
        self.total_dict[trial] = save_dict

        # save data
        if time.perf_counter() - self.time_every_thirty_seconds > 10:
            print('Saving copy...')
            # save data when experiment finishes
            with open("../data/" + experiment['name'] + "/sample.json", "w") as outfile:
                json.dump(self.total_dict, outfile, indent=4)
            self.time_every_thirty_seconds = time.perf_counter()

    def run(self):
        global activate_countdown, show_green, endTrial, sevenSecondStop, countdown, \
            time_beginning_trial, trial, width, height, finger_number, finger_number_list, experiment
        # Start the hand tracking using the HAND model
        count_frames_hand_not_detected = 0
        directory_video = '../data/' + experiment['name'] + '/'
        video_result = cv2.VideoWriter(directory_video + '1.avi', 
                                       cv2.VideoWriter_fourcc(*'MJPG'),
                                       20.0, 
                                       (width, height))
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.25,
                min_tracking_confidence=0.25,
                max_num_hands=2) as hands:
            while cap.isOpened():
                success, self.original_image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Apply mediapipe HAND detection model
                self.image, results = self.mediapipe_detection(self.original_image, hands)

                detected_hands = []
                if results.multi_hand_landmarks:  # this is a list with [landmarks_hand1, landmarks_hand2]
                    self.LR = []  # list that will contain both landmarks from left and right hands
                    for rl, hand_landmarks in enumerate(
                            results.multi_hand_landmarks):  # iterate through all the detected hands

                        # Draw the hand landmarks
                        self.draw_landmarks(self.image, hand_landmarks)

                        # Get hand index to check label (left or right)
                        hand_index = results.multi_hand_landmarks.index(hand_landmarks)
                        hand_score = results.multi_handedness[hand_index].classification[0].score
                        hand_label = results.multi_handedness[hand_index].classification[0].label

                        # Flip right and left hand because the image is mirrored
                        hand_label = 'Right' if hand_label == 'Left' else 'Left'
                        detected_hands.append(hand_label)
                        # Initialize a list with the hand label and its score
                        hand_landmarks_data = [[hand_label, hand_score]]

                        # Initialize a list with the hand landmarks, which will be used to create a black mask
                        contours = []
                        for landmarks in hand_landmarks.landmark:
                            # Append the hand_landmarks_data list with x, y and z positions of each landmark
                            # x coordinates are flipped, so we must do [1-landmarks.x]
                            hand_landmarks_data.append([int(width*(1-landmarks.x)), int(height*(landmarks.y)), int(width*(landmarks.z))])
                            # Append the contours list with x, y coords. of the right hand considering screen dimensions
                            if hand_label == 'Left':
                                contours.append([int(landmarks.x * width), int(landmarks.y * height)])
                                # print('right hand detected')

                        #if contours:
                            #self.image = self.hide(self.image, contours)

                        # Append to the LR list to save the information from both hands
                        self.LR.append(hand_landmarks_data)

                    # Right hand is appended first to the list, then the left hand
                    if len(self.LR) == 2 and self.LR[0][0][0] == 'Left':
                        self.LR[0], self.LR[1] = self.LR[1], self.LR[0]

                    # Assess distances [hand][index finger landmark 8][coordinate]
                    # if left and right hands have been detected and both hands are different side
                    if len(self.LR) == 2 and self.LR[0][0][0] != self.LR[1][0][0]:
                        # Get the index fingertip coordinates from the right hand
                        #id1 = 0 if self.LR[0][0][0] == 'Right' else 1
                        #id2 = 1 if self.LR[0][0][0] == 'Right' else 0

                        # Get the index fingertip coordinates from the right hand
                        self.r_idx_8 = self.LR[0][8 + 1]

                        # Get all fingertip coordinates from the left hand
                        self.l_idx_list = [self.LR[1][4 + 1], self.LR[1][8 + 1], self.LR[1][12 + 1],
                                           self.LR[1][16 + 1], self.LR[1][20 + 1]]

                        # Get the target fingertip coordinates from the left hand
                        self.l_idx = self.l_idx_list[finger_number]
                        self.finger_coordinates = self.finger_coordinates_list[finger_number]

                        # when inside the green circle, activate the countdown (if green dot is displayed)
                        if width - 60 * wfactor - 30 * wfactor < self.r_idx_8[0] < width - 60 * wfactor + 30 * wfactor \
                                and height / 2 - 30 * wfactor < self.r_idx_8[1] < height / 2 + 30 * wfactor:
                            if show_green:
                                activate_countdown = True
                        else:
                            activate_countdown = False

                        # Euclidean distances considering xy coordinate
                        self.dist_xy = int((((self.r_idx_8[0] - self.l_idx[0]) ** 2)
                                           + (self.r_idx_8[1] - self.l_idx[1]) ** 2) ** (1 / 2))

                        if countdown == 0:
                            # add frames to video to be saved
                            video_result.write(cv2.flip(self.original_image, 1))

                            # Append to lists which contain every frame information for this trial
                            self.all_coord.append(self.LR)
                            self.all_distances.append(self.dist_xy)
                            self.all_times.append(round(time.perf_counter() - time_beginning_trial, 4))

                            # Append the xy distances to a list to know when the user has stopped moving his fingers
                            # Save value in a rolling method, keeping the old values in the same position
                            self.my_array[self.current_idx] = self.dist_xy
                            self.current_idx = (self.current_idx+1) % self.max_size

                            # If the index fingertips are close dist_xy < 60
                            # and have stopped moving std < 100, start the next trial
                            if all(elem is not None for elem in self.my_array):
                                self.std = int(statistics.stdev(self.my_array))
                                # print(self.dist_xy, self.std)
                            endTrial = True if (self.std < 100 and self.dist_xy < 60) else False

                            # If endTrial is true, then the trial has ended and the frame can be saved
                            # This is the final frame - Save it now (if the users are not correcting the trajectories)

                        # When the trial has been finished, empty the following lists 
                        if countdown == 3:
                            self.all_coord = []
                            self.all_distances = []
                            self.all_times = []

                        # Variable to display information about a hand not being detected,
                        # but if it has not been detected for more than 20 frames
                        count_frames_hand_not_detected = 0
                    else:
                        count_frames_hand_not_detected += 1
                        if 'Left' not in detected_hands:
                            detection_warning_text = 'Left hand NOT detected'
                        elif 'Right' not in detected_hands:
                            detection_warning_text = 'Right hand NOT detected'
                else:
                    count_frames_hand_not_detected += 1
                    detection_warning_text = 'NO HANDS DETECTED'

                # Flip the image because it should be mirrored
                self.image = cv2.flip(self.image, 1)

                # If the hands are not being detected for more than 20 frames, display a reminder to show both hands
                if count_frames_hand_not_detected > 20:
                    self.image = cv2.rectangle(self.image, (20, height - 80), (width - 20, height - 20),
                                               (255, 255, 255), -1)
                    self.image = cv2.putText(self.image, detection_warning_text,
                                             (30, height - 40), cv2.FONT_HERSHEY_SIMPLEX,
                                             0.7 * wfactor, (0, 0, 255), 2, cv2.LINE_AA)

                # Display how many trials are left in a corner of the screen
                self.image = cv2.putText(self.image, str(trial) + '/30',
                                         (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                         0.7 * wfactor, (255, 255, 255), 2, cv2.LINE_AA)

                # Display the cartoon hand image to know where the user has to move
                hand_img = cv2.imread('../assets/hand_rotated.png')
                hand_img = cv2.resize(hand_img, (int(190 * wfactor), int(150 * hfactor)), interpolation=cv2.INTER_AREA)
                # Add hand_image on top of the video
                self.image[10:10 + int(150 * hfactor),
                           width - int(190 * wfactor) - 10:width - 10] = hand_img

                # Plot red circle on top of the desired finger. Plot finger coordinates
                self.image = cv2.circle(self.image, self.finger_coordinates, 15, (0, 0, 255), -1)

                # Plot the green circle
                if show_green:
                    self.image = cv2.circle(self.image, (int(width - 60 * wfactor), int(height / 2)), int(30 * wfactor),
                                            (0, 255, 0), -1)
                    # Put text image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
                    self.display_countdown = 'GO' if countdown == 0 else countdown
                    self.image = cv2.putText(self.image, str(self.display_countdown),
                                             (int(width - 60 * wfactor - 10), int(height / 2 + 10)),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                
                # If the trial has finished or the user has run out of time (7 seconds max)
                if endTrial or sevenSecondStop:
                    # Save data for that trial
                    self.save_data()

                    # Change the target finger
                    finger_number_list.remove(finger_number)
                    # If the list has emptied, then populate it again with numbers
                    if not finger_number_list:
                        finger_number_list = [0, 1, 2, 3, 4]
                    # Randomly select the next target finger based on the available fingers (finger_number_list)
                    finger_number = random.choice(finger_number_list)

                    # When finished, save the video with all the frames
                    video_result.release()
                    # Save the frame
                    cv2.imwrite('../data/' + experiment['name'] + '/' + str(trial)+'.JPG', cv2.flip(self.original_image, 1))

                    # Prepare video writer for the next trial
                    video_result = cv2.VideoWriter(directory_video + str(trial + 1) + '.avi',
                                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                                   20.0, 
                                                   (width, height))
                
                if countdown == 0:
                    # If the user has run out of time (7 seconds max) then display a warning message
                    if sevenSecondStop:
                        cv2.imshow("Image", self.seven_second_warning)
                        cv2.waitKey(2000)
                        sevenSecondStop = False
                        countdown = 3
                        show_green = True
    
                    # If the trial has ended in time then display a message informing the user to go back to the start
                    if endTrial:
                        next_img = np.zeros((height, width, 3), np.uint8)
                        next_img.fill(255)
                        end_trial_image = cv2.putText(next_img,
                                                      'Next trial, go to the Green dot', (30, height - 40),
                                                      cv2.FONT_HERSHEY_SIMPLEX, 1 * wfactor, (51, 153, 0), 2, cv2.LINE_AA)
                        cv2.imshow("Image", end_trial_image)
                        cv2.waitKey(1000)
                        endTrial = False
                        show_green = True
                        countdown = 3
    
                # Finish execution after 30 trials automatically
                if trial >= 30:
                    print('You did a total of {} trials'.format(trial))
                    # save data when experiment finishes
                    with open("../data/" + experiment['name'] + "/sample.json", "w") as outfile:
                        json.dump(self.total_dict, outfile, indent=4)

                    # release video capture and video writer
                    cap.release()
                    video_result.release()

                    # Stop execution
                    os._exit(0)

                # Show the video image with all the variations
                cv2.imshow("Image", self.image)

                k = cv2.waitKey(1)
                if k == 27:
                    cv2.destroyWindow('Image')
                    os._exit(0)


if __name__ == "__main__":
    print('Start run')

    # Start background threads
    countdown_thread = Thread(target=countdown_task)
    countdown_thread.start()
    trialTimeLimit_thread = Thread(target=trial_time_limit_task)
    trialTimeLimit_thread.start()

    # Initialize global variables
    activate_countdown = False  # The countdown activates when the user has been on the green button for 3 seconds
    show_green = True           # When the task has ended, the green button will be displayed again
    endTrial = False            # When the user stops moving, the endTrial will be set to True
    sevenSecondStop = False     # The user has 7 seconds to perform the task, or it will jump to the next one
    countdown = 3               # The user will have to remain on the green dot for 3 seconds
    trial = 0                   # Initlaize trial counter
    width, height = 1280, 720   # Specify the resolution from the camera
    wfactor, hfactor = 1.6, 1.6  # 1280 / 720, # 720 / 480 # resolution resizing parameters
    finger_number_list = [0, 1, 2, 3, 4]                # A list with the 5 fingers [thumb, index, middle, ring, pinkie]
    finger_number = random.choice(finger_number_list)   # The target finger is randomized
    right_finger_indeces = [5, 9, 13, 17, 21]           # The Mediapipe indices from the 5 fingers
    inp = None                                          # for the name input box

    # Call the input class to define the name of the user
    name_input = Input()
    experiment = name_input.run()

    # Initialize counter at the beginning of the assessment
    time_beginning_trial = time.perf_counter()

    # Capture a webcam video and define its resolution
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Check if cameras opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    # Define an instance of the App()
    app = App()

    # Start the main thread (the main code)
    main_thread = Thread(target=app.run)
    main_thread.start()
