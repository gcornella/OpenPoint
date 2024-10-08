"""
-------------------------------------------------------------------------------
 Title        : hand_size_calibration.py
 Description  : Input an image of a hand on a sheet of paper to calculate the hand size in cm
 Author       : Guillem Cornella
 Date Created : February 2024
-------------------------------------------------------------------------------
"""

import cv2
import numpy as np
from skimage.measure import label, regionprops
import mediapipe as mp
import json
import sys

# Import mediapipe utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def mediapipe_detection(imag, model):
    # Function used to enhance the mediapipe detection
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
    # Function to draw the landmarks using a specific style
    mp_drawing.draw_landmarks(
        im,
        result,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

def calculate_hand_size(hand_coordinates, w, h):
    # To normalize the error distance at every frame so that the depth of the hand does not affect the results,
    # calculate the distance between some hand landmark to have a standard hand size in pixels (px) for every user.
    # Distance calculated using the Euclidean distance in pixels according to the camera's resolution (w x h)
    if landmark_joints == 'fingertips':
        s = [8, 12, 16, 20, 4, 5, 17]   # starting landmark indices
        e = [5, 9, 13, 17, 2, 0, 0]     # ending landmark indices
    elif landmark_joints == 'MCP':
        s = [5, 17, 17, 2]  # starting landmark indices change for MCP
        e = [0, 0, 5, 0]  # ending landmark indices change for MCP
    elif landmark_joints == 'PIP':
        s = [6, 10, 14, 18, 2, 5, 17]  # starting landmark indices change for PIP
        e = [5, 9, 13, 17, 0, 0, 0]  # ending landmark indices change for PIP
    else:
        print('Choose an appropriate landmark_joints variable name.')
        sys.exit("Strings are not equal, exiting.")

    hand_size_px = 0
    for i in range(0, len(s)):
        # Calculate the distance in pixels for every pair of lines between the selected indices
        line_length_px = np.sqrt(((hand_coordinates[s[i]][0] - hand_coordinates[e[i]][0]) * w) ** 2
                                 + ((hand_coordinates[s[i]][1] - hand_coordinates[e[i]][1]) * h) ** 2)
        hand_size_px += line_length_px

    return hand_size_px


# if the user decides to take a photo in realtime
def upload_from_video():
    width = 1280
    height = 720
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Insert the corners info image
    calib_corners = cv2.imread('calib_corners.png')
    calib_corners = cv2.resize(calib_corners, (width, height))

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
    else:
        # Capture frames from the camera
        while True:
            ret, frame = cap.read()
            if ret:
                blend = cv2.addWeighted(frame, 1, calib_corners, 0.3, 0)

                # Display the captured frame
                cv2.imshow('Frame', blend)
                # Break the loop if 'q' is pressed
                key = cv2.waitKey(1)
                if key & 0xFF == ord('s'):
                    print("Letter 's' pressed!")
                    cv2.imwrite('capture.jpg', frame)
                    break
                # Break the loop if 'q' is pressed
                elif key & 0xFF == ord('q'):
                    break
            else:
                print("Error: Could not read frame.")
                break
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()
    return frame


def apply_mediapipe(image, paper_perimeter_px, cut):
    print(" --- Applying mediapipe to the image ---")
    # This functions uses the Mediapipe hand detection model to estimate hand landmarks
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.25,
            min_tracking_confidence=0.25,
            max_num_hands=1) as hands:

        # Apply Mediapipe HAND detection model
        image, results = mediapipe_detection(image, hands)

        LR = []  # list that will contain both landmarks from left and right hands
        if results.multi_hand_landmarks:  # this is a list with [landmarks_hand1, landmarks_hand2]

            for rl, hand_landmarks in enumerate(
                    results.multi_hand_landmarks):  # iterate through all the detected hands

                # Draw the hand landmarks
                draw_landmarks(image, hand_landmarks)

                # Get hand index to check label (left or right)
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handScore = results.multi_handedness[handIndex].classification[0].score
                handLabel = results.multi_handedness[handIndex].classification[0].label

                # Flip right and left hand because the image is mirrored
                # handLabel = 'Right' if handLabel == 'Left' else 'Left'

                # Initialize a list with the hand label and its score
                handLandmarks = [[handLabel, handScore]]

                for landmarks in hand_landmarks.landmark:
                    # Append the handLandmarks list with x, y and z positions of each landmark
                    handLandmarks.append([landmarks.x, landmarks.y, landmarks.z])
                LR.append(handLandmarks)

        # Add data to the dictionary if LR not empty
        if LR:
            my_dict = {}
            my_dict['user'] = user_id

            # Call the calculate_hand_size() function to get the dimension of the user's hand in pixels
            hand_coordinates = LR[0][1:] # either the left or the right hand
            hand_size_px = calculate_hand_size(hand_coordinates, w, h)
            my_dict['hand_size_px'] = hand_size_px

            # Size of a letter paper is
            paper_width = 27.94
            paper_height = 21.59
            paper_perimeter_cm = 2 * paper_width + 2 * paper_height
            paper_perimeter_px_entirewindow = 2 * w + 2 * h
            print('paper perimeter of the uncut image is {} pixels'.format(paper_perimeter_px))
            print('paper perimeter occupying the entire screen is {} pixels'.format(paper_perimeter_px_entirewindow))

            # Convert from cm to pixels
            scale_cmpx = paper_perimeter_px / paper_perimeter_cm
            print("1 cm equals to: ", scale_cmpx * 1, " pxs")
            print("paper width (cm) equals to: ", scale_cmpx * 27.94, " pxs")

            # Convert from pixels to cm
            scale_pxcm = paper_perimeter_cm / paper_perimeter_px
            print("hand size (px) equals to: ", scale_pxcm * hand_size_px, " cms")
            my_dict['hand_size_cm'] = scale_pxcm * hand_size_px

            with open(user_id+'_calib_new'+ '_cut' + str(cut) + '.json', 'w') as file:
                json.dump(my_dict, file)

    return image


def extract_corners(img):
    try:
        # Start with morphological operations to get a blank page.
        # Repeated Closing operation to remove text from the sheet of paper.
        kernel = np.ones((10,10),np.uint8)
        img_morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 1)

        # Convert the input image into the grayscale color space
        operated_image = cv2.cvtColor(img_morph, cv2.COLOR_BGR2GRAY)
        # Modify the data type setting to 32-bit floating point
        operated_image = np.float32(operated_image)
        # Apply the cv2.cornerHarris method to detect the corners
        dest = cv2.cornerHarris(operated_image,     # image
                                10,                 # blockSize (size of neighbourhood)
                                5,                  # ksize (aperture parameter of the Sobel used)
                                0.07)               # k (free parameter)

        # Dilate the corners for better visualization and detection of corner centroids
        kernel = np.ones((10,10), np.uint8)
        dest = cv2.dilate(dest, kernel)

        # Reverting back to the original image to visualize the corners in red, with optimal threshold value
        img_visualization = img_morph.copy()
        # Change this threshold to find a better corner detection. The higher it is, the more restrictive
        threshold = 0.1
        bool_array = dest > threshold * dest.max()
        img_visualization[bool_array] = [0, 0, 255]
        cv2.imshow('Corners', img_visualization)
        # cv2.waitKey(0)

        # Get the centroid of the detected corner or blob
        labelled_array = label(bool_array)
        # Get the region properties of each blob using skimage library
        regions = regionprops(labelled_array)

        # Calculate the centroid of each blob, iterating through all the detected blobs
        centroids = []
        for region in regions:
            centroid = region.centroid
            h_centr = int(centroid[0])
            w_centr = int(centroid[1])
            # print('hcent: ', h_centr, 'w_centr: ',w_centr)
            # If in the borders of the image (not in the center of the screen) then save point as a corner
            if (10 < h_centr < h/2 or h-10 > h_centr > 2/4*h) and (10 < w_centr < w/2 or w-10 >w_centr > 2/4*w):
                centroids.append([w_centr, h_centr])

        # Get the 4 corners, taking mins and maxs of centroids list
        # Find the minimum and maximum x and y coordinates
        min_x = min(coord[0] for coord in centroids)
        max_x = max(coord[0] for coord in centroids)
        min_y = min(coord[1] for coord in centroids)
        max_y = max(coord[1] for coord in centroids)

        # Calculate distances from corners
        distances = [
            (coord[0] - min_x) ** 2 + (coord[1] - min_y) ** 2 +
            (coord[0] - max_x) ** 2 + (coord[1] - min_y) ** 2 +
            (coord[0] - min_x) ** 2 + (coord[1] - max_y) ** 2 +
            (coord[0] - max_x) ** 2 + (coord[1] - max_y) ** 2
            for coord in centroids
        ]

        # Get indices of four closest points
        indices = sorted(range(len(distances)), key=lambda i: distances[i], reverse=True)[:4]

        # Get the four extreme coordinates
        coordinates = [centroids[i] for i in indices]

        up_left = []
        up_right = []
        low_left = []
        low_right = []

        # Determine the center of the coordinate system
        center_x = sum(x for x, y in coordinates) / len(coordinates)
        center_y = sum(y for x, y in coordinates) / len(coordinates)

        for x, y in coordinates:
            if x < center_x and y < center_y:
                up_left.append([x, y])
            elif x >= center_x and y < center_y:
                up_right.append([x, y])
            elif x < center_x and y >= center_y:
                low_left.append([x, y])
            elif x >= center_x and y >= center_y:
                low_right.append([x, y])

        print("Up Left:", up_left)
        print("Up Right:", up_right)
        print("Low Left:", low_left)
        print("Low Right:", low_right)
        centroids = [up_left, up_right, low_left, low_right]

        # Define the four corners of the original image
        corners = np.float32(centroids)

        paper_perimeter_px = np.sqrt((up_left[0][0] - up_right[0][0]) ** 2 + (up_left[0][1] - up_right[0][1]) ** 2) + \
                             np.sqrt((up_left[0][0] - low_left[0][0]) ** 2 + (up_left[0][1] - low_left[0][1]) ** 2) + \
                             np.sqrt((low_left[0][0] - low_right[0][0]) ** 2 + (low_left[0][1] - low_right[0][1]) ** 2) + \
                             np.sqrt((up_right[0][0] - low_right[0][0]) ** 2 + (up_right[0][1] - low_right[0][1]) ** 2)

        # Call the mediapipe hand detection model before cutting the image using the corners.
        image = apply_mediapipe(img, paper_perimeter_px, False)
        cv2.imshow('Result without cutting', image)

        # Define the four corners of the original image
        destination_corners = np.float32([[0, 0], [w, 0],[0, h], [w, h]])

        # Compute the perspective transform matrix
        if len(corners)==4:
            matrix = cv2.getPerspectiveTransform(corners, destination_corners)
            # Apply the perspective transformation to the image
            result = cv2.warpPerspective(img, matrix, (w, h))

        # Once the perspective transformation has been applied, call mediapipe again to calculate hand size
        paper_perimeter_px = w * 2 + h * 2

        # Apply mediapipe again with the hand occupying the entire screen
        final_image = apply_mediapipe(result, paper_perimeter_px, True)

        return final_image

    except ValueError:
        print('Upload another image with visible margins')
        #break


# Define the main block
if __name__ == "__main__":

    # Define the user's id or the file_path of your calibration image
    user_id = 'test_user'      # add your user_id here
    print('Running hand size normalization for {}'.format(user_id))
    
    # Depending on the landmarks you want to interact with (MCP, PIP, or fingertips), select an appropriate function
    # After this calibration you'll calculate the error from the one index fingertip against each contralateral landmark
    landmark_joints = 'fingertips'  # can also be 'MCP', or 'PIP'
    print('Calculating distance considering {} landmarks'.format(landmark_joints))
    
    # Choose if you want to upload an image or to take a picture from the webcam
    # Place your hand on top on a letter-sized blank paper on a desk, and avoid having obstacles, then take a picture.
    uploadMethod = 'file'      # alternative is 'webcam' in real rime.
    # Press 's' to save image in the 'webcam' scenario
    print('Reading image from {}'.format(uploadMethod))
    
    if uploadMethod == 'file':
        # Read the given image from the hand on a sheet of paper
        img = cv2.imread(user_id+'.JPG')

        # Prepare the image for the corner extraction
        # Get the dimensions of the image [height, width, channels]
        h, w, c = np.shape(img)
        print('Height:', h, ' - Width:', w, ' - Channels:', c)

        # If the image is uploaded vertically then rotate it. Using aspect ratio
        if w / h < 1:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Get the dimensions of the rotated image [height, width, channels]
            h, w, c = np.shape(img)

        # Resize the image for standardization if needed. Comment the following lines if you don't want to resize
        w, h = int(w / 4), int(h / 4)
        img = cv2.resize(img, (w, h))
        print('Image resized size: ', img.shape)

    else:
        # Upload from a webcam's video (not recommended)
        w, h = 1280, 720
        img = upload_from_video()

    # Detect the corners of the sheet of paper (with known size), to standardize the hand size
    # This will be used to normalize the hand to its depth for better precision.
    image_cut = extract_corners(img)

    cv2.imshow('Result cut image', image_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
