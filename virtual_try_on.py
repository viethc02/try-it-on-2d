import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the cloth image
cloth = cv2.imread('Images/tshirt4.jpg')

# Initialize video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image color to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    frame.flags.writeable = False

    # Make detection
    results = pose.process(frame)

    # To improve performance, optionally mark the image as writeable to pass by reference.
    frame.flags.writeable = True

    # Convert the image color back to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks is not None:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Get the coordinates of landmarks 11, 12, 23, 24
        pose_coords = [[landmarks[11].x, landmarks[11].y],
                       [landmarks[12].x, landmarks[12].y],
                       [landmarks[23].x, landmarks[23].y],
                       [landmarks[24].x, landmarks[24].y]]

        # Draw landmarks on the image
        for landmark in landmarks:
            
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            if landmark == landmarks[11] or landmark == landmarks[12] or landmark == landmarks[23] or landmark == landmarks[24] :
                cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)
                continue
           # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
	
        imgshirt = cv2.imread('Images/tshirt4.jpg')
        musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)  # grayscale conversion
        ret, orig_mask = cv2.threshold(musgray, 200, 255, cv2.THRESH_BINARY)
        orig_mask_inv = cv2.bitwise_not(orig_mask)

        # Add a cloth at the landmark coordinates
        cloth_height = int(frame.shape[0] * abs(pose_coords[3][1] - pose_coords[0][1]))
        cloth_width = int(frame.shape[1] * abs(pose_coords[1][0] - pose_coords[0][0]))

        if cloth_height > 0 and cloth_width > 0 and pose_coords[1][1] < 1 and pose_coords[1][0] < 1 and pose_coords[3][0] < 1 and pose_coords[3][1] < 1:
            shirt_y1 = int(pose_coords[1][1] * frame.shape[0]) - 50
            shirt_y2 = int(pose_coords[1][1] * frame.shape[0]) + cloth_height 
            shirt_x1 = int(pose_coords[1][0] * frame.shape[1]) - 50
            shirt_x2 = int(pose_coords[1][0] * frame.shape[1]) + cloth_width + 50

            if shirt_x1 < 0:
                shirt_x1 = 0
            if shirt_y1 < 0:
                shirt_y1 = 0
            if shirt_x2 > frame.shape[1]:
                shirt_x2 = frame.shape[1]
            if shirt_y2 > frame.shape[0]:
                shirt_y2 = frame.shape[0]

            shirtWidth = shirt_x2 - shirt_x1
            shirtHeight = shirt_y2 - shirt_y1
            if shirtWidth < 0 or shirtHeight < 0:
                continue
            shirtWidth = shirtWidth 
            shirtHeight = shirtHeight 
            shirt = cv2.resize(imgshirt, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)

            # take ROI for shirt from background equal to size of shirt image
            roi = frame[shirt_y1 :shirt_y2 , shirt_x1 :shirt_x2]
            print(roi.shape)
            print(mask.shape)
            # roi_bg contains the original image only where the shirt is not
            # in the region that is the size of the shirt.
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
            roi_fg = cv2.bitwise_and(shirt, shirt, mask=mask_inv)
            dst = cv2.add(roi_bg, roi_fg)
            
            frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = dst
        else: 
            # get position of 
            shirt_y1 = int(pose_coords[1][1] * frame.shape[0]) - 60
            shirt_y2 = int(pose_coords[1][1] * frame.shape[0]) + cloth_height 
            shirt_x1 = int(pose_coords[1][0] * frame.shape[1]) - 50
            shirt_x2 = int(pose_coords[1][0] * frame.shape[1]) + cloth_width + 50
            
            old_x2 = shirt_x2
            old_y2 = shirt_y2
            if shirt_x1 < 0:
                shirt_x1 = 0
            if shirt_y1 < 0:
                shirt_y1 = 0
            if shirt_x2 > frame.shape[1]:
                shirt_x2 = frame.shape[1]
            if shirt_y2 > frame.shape[0]:
                shirt_y2 = frame.shape[0]

            shirtWidth = shirt_x2 - shirt_x1
            shirtHeight = shirt_y2 - shirt_y1

            old_shirtWidth = old_x2 - shirt_x1
            old_shirtHeight = old_y2 - shirt_y1
            if shirtWidth < 0 or shirtHeight < 0:
                continue

            # get ratio from shoulder to body wit 
            ratio_width = shirtWidth / old_shirtWidth
            ratio_height = shirtHeight / old_shirtHeight

            imgshirt = imgshirt[0:int(480*ratio_height), 0:int(640*ratio_width)]

            musgray = cv2.cvtColor(imgshirt, cv2.COLOR_BGR2GRAY)  # grayscale conversion
            ret, orig_mask = cv2.threshold(musgray, 200, 255, cv2.THRESH_BINARY)
            orig_mask_inv = cv2.bitwise_not(orig_mask)

            #resize shirt image and mask according to width and height of body
            shirt = cv2.resize(imgshirt, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (shirtWidth, shirtHeight), interpolation=cv2.INTER_AREA)

            # take ROI for shirt from background equal to size of shirt image
            roi = frame[shirt_y1 :shirt_y2 , shirt_x1 :shirt_x2]
            print(roi.shape)
            print(mask.shape)
            # roi_bg contains the original image only where the shirt is not
            # in the region that is the size of the shirt.
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
            roi_fg = cv2.bitwise_and(shirt, shirt, mask=mask_inv)
            dst = cv2.add(roi_bg, roi_fg)
            
            frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2] = dst
    # Display the resulting image
    cv2.imshow('Try On Clothes', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()