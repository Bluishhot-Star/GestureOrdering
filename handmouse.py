import cv2  
import mediapipe as mp   
from PIL import ImageFont, ImageDraw, Image 
import numpy as np
import pyautogui   

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


screen_width, screen_height = pyautogui.size()
print(screen_width, screen_height)

pre = ''
offset = 200
grab = False

cap = cv2.VideoCapture(0)


with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
  while cap.isOpened():
    success, image = cap.read()

    h,w,c = image.shape

    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)

    # screen_height : y_ = image_height : y
    # y_ = y*screen_height/image_height₩

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        thumb_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height:
              thumb_finger_state = 1

        index_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
              index_finger_state = 1

        middle_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
              middle_finger_state = 1

        ring_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
              ring_finger_state = 1

        pinky_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height:
          if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height:
              pinky_finger_state = 1

        
        
        # 이미지 좌표 (검지 뿌리)
        image_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x*image_width
        image_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y*image_height

        image_x = image_x - offset
        image_y = image_y - offset
        new_image_height = image_height - offset*2.5
        new_image_width = image_width - offset*2.5
        screen_y = image_y*screen_height/new_image_height
        screen_x = image_x*screen_width/new_image_width

        pyautogui.moveTo(screen_x, screen_y)
        pre = 'move'
        # if image_x > offset and image_x < w-offset and image_y > offset and image_y < h - offset:
        #     image_x = image_x - offset
        #     image_y = image_y - offset
        #     new_image_height = image_height - offset*2.5
        #     new_image_width = image_width - offset*2.5

        #     screen_y = image_y*screen_height/new_image_height
        #     screen_x = image_x*screen_width/new_image_width

        #     pyautogui.moveTo(screen_x, screen_y)
        #     pre = 'move'

        # 주먹 쥐었을때
        if index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
            grab = True
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            pre = 'left click'

        print(grab)

        # 손가락 위치 확인한 값을 사용하여 가위,바위,보 중 하나를 출력 해줍니다.
        font = ImageFont.load_default()
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        w, h = font.getsize(pre)

        x = 50
        y = 50

        draw.rectangle((x, y, x + w, y + h), fill='black')
        draw.text((x, y),  pre, font=font, fill=(255, 255, 255))
        image = np.array(image)

        cv2.rectangle(image, (offset, offset), (image_width-offset, image_height-offset), (255,0,0), 2)

        # 손가락 뼈대를 그려줍니다.
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()