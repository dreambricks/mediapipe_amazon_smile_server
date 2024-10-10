import pickle

import cv2

from utils import FaceLandmarks

import keyboard

draw_mask = False
draw_stats = True
terminate_program = False

# Callback function to be triggered on key press
def on_key_event(event):
    global draw_mask, draw_stats
    print(f"Key {event.name} was pressed")
    if event.name == 'm':
        draw_mask = not draw_mask
    elif event.name == 's':
        draw_stats = not draw_stats
    elif event.name == 'q':
        terminate_program = True

# Hook the key event
keyboard.on_press(on_key_event)

fl = FaceLandmarks(static_image_mode=False)
emotions = ['happy', 'laughing', 'neutral'] # , 'smiling'
emotions_pt = ['sorrindo', 'rindo', 'neutro'] # , 'sor. leve'
emotions_idx_sorted = [1, 2, 0]
with open('./model4', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while ret:
    #if terminate_program:
    #    break
    ret, frame = cap.read()

    face_landmarks = fl.get_face_landmarks(frame, draw=draw_mask)
    
    if len(face_landmarks) == 1404:
        #output = model.predict([face_landmarks])
        output = model.predict_proba([face_landmarks])
        #print(output[0])
        if draw_stats:
            max_val = max(output[0])
            for idx, e_idx in enumerate(emotions_idx_sorted):
                e = emotions_pt[idx]
                text = f"{e} : {output[0][idx]*100:.0f}%"
                color = (0, 255, 0) if output[0][idx] == max_val else (0, 0, 255)
                cv2.putText(frame,
                            text,
                           (10, frame.shape[0] - 10 - (len(emotions)-e_idx-1)*35),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1,
                           color,
                           3)
            
    cv2.imshow('frame', frame)

    cv2.waitKey(25)


cap.release()
cv2.destroyAllWindows()