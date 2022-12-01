import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['select', 'next', 'exit']
seq_length = 30

model = load_model('models/model2_1.0.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)

# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
# out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break


[x: 0.37395426630973816
y: 0.9835489988327026
z: 6.124909024407543e-09
, x: 0.33027780055999756
y: 0.9101158976554871
z: 0.07722993195056915
, x: 0.301399827003479
y: 0.8385507464408875
z: 0.10229119658470154
, x: 0.2795636057853699
y: 0.7904409170150757
z: 0.11024373024702072
, x: 0.25802963972091675
y: 0.7624066472053528
z: 0.11849714815616608
, x: 0.31567779183387756
y: 0.7424973249435425
z: 0.0706661269068718
, x: 0.2529051899909973
y: 0.6775016784667969
z: 0.08561907708644867
, x: 0.23013794422149658
y: 0.7266045808792114
z: 0.10310465097427368
, x: 0.23238293826580048
y: 0.7853365540504456
z: 0.11458799242973328
, x: 0.3138679265975952
y: 0.747907817363739
z: 0.027138153091073036
, x: 0.24406953155994415
y: 0.6889314651489258
z: 0.03189653530716896
, x: 0.22262832522392273
y: 0.7436988949775696
z: 0.04089755564928055
, x: 0.22678984701633453
y: 0.7946879863739014
z: 0.04991519823670387
, x: 0.3054477870464325
y: 0.7669987082481384
z: -0.010237718001008034
, x: 0.23654210567474365
y: 0.7089377641677856
z: -0.0022928027901798487
, x: 0.21842406690120697
y: 0.7645960450172424
z: 0.013662565499544144
, x: 0.2260826975107193
y: 0.8152028322219849
z: 0.024279678240418434
, x: 0.29483693838119507
y: 0.7960667014122009
z: -0.042312175035476685
, x: 0.23663847148418427
y: 0.7478399276733398
z: -0.031433623284101486
, x: 0.22245770692825317
y: 0.7854004502296448
z: -0.015154696069657803
, x: 0.22778700292110443
y: 0.8303093910217285
z: -0.0038070729933679104
]