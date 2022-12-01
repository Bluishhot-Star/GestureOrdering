import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue


    # OpenCV = BGR, Mediapipe = RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 프레임 당 BGR-->RGB
    results1 = hands.process(image) # 전처리된 이미지 저장
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # 화면에 출력위해 RGB-->BGR
    if results1.multi_hand_landmarks:
      for hand_landmarks in results1.multi_hand_landmarks:
        # print(hand_landmarks)
        # print(hand_landmarks.landmark)
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        distance = int(abs(thumb.x-index.x)*300)
        cart = 3
        if(distance == 0):
            grab = True
        else:
            grab = False
        # print(f"그랩 : {grab}")
        
        print(f"엄지: {thumb} 검지: {index}")

        # 우측 상단 0,0

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    # if cv2.waitKey(5) & 0xFF == 27:
    #   break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results2 = pose.process(image)

    # Draw the pose annotation on the image2.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results2.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image2 horizontally for a selfie-view display.
    # img = np.ones((480, 640, 3), dtype=np.uint8) * 255 --> 하얀색 
    cv2.imshow('MediaPipe Pose', cv2.flip(img, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()