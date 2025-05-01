
# import cv2 as cv
# import mediapipe as mp
# import time
# import utils, math
# import numpy as np
# import pygame
# pygame.init()
# pygame.mixer.init()


# frame_counter =0
# CLOSED_EYES_FRAME =3
# FONTS =cv.FONT_HERSHEY_COMPLEX
# wait_time = 1
# sound = pygame.mixer.Sound("drowsy alarm.wav")

# check_interval = 0.1
# num_iterations = int(wait_time / check_interval)

# FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
# LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
# UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
# RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# map_face_mesh = mp.solutions.face_mesh
# camera = cv.VideoCapture(0)
# def landmarksDetection(img, results, draw=False):
#     img_height, img_width= img.shape[:2]
#     mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
#     if draw :
#         [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

#     return mesh_coord

# def euclaideanDistance(point, point1):
#     x, y = point
#     x1, y1 = point1
#     distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
#     return distance

# def blinkRatio(img, landmarks, right_indices, left_indices):
#     rh_right = landmarks[right_indices[0]]
#     rh_left = landmarks[right_indices[8]]
#     rv_top = landmarks[right_indices[12]]
#     rv_bottom = landmarks[right_indices[4]]
#     lh_right = landmarks[left_indices[0]]
#     lh_left = landmarks[left_indices[8]]

#     lv_top = landmarks[left_indices[12]]
#     lv_bottom = landmarks[left_indices[4]]

#     rhDistance = euclaideanDistance(rh_right, rh_left)
#     rvDistance = euclaideanDistance(rv_top, rv_bottom)

#     lvDistance = euclaideanDistance(lv_top, lv_bottom)
#     lhDistance = euclaideanDistance(lh_right, lh_left)

#     reRatio = rhDistance/rvDistance
#     leRatio = lhDistance/lvDistance

#     ratio = (reRatio+leRatio)/2
#     return ratio 



# with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

#     start_time = time.time()
#     while True:
#         frame_counter +=1
#         ret, frame = camera.read()
#         if not ret: 
#             break 

#         frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
#         frame_height, frame_width= frame.shape[:2]
#         rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
#         results  = face_mesh.process(rgb_frame)
#         if results.multi_face_landmarks:
#             mesh_coords = landmarksDetection(frame, results, False)
#             ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
#             elapsed_time = 0
#             while elapsed_time <= wait_time:
#                 start_time = time.time()

#                 if ratio < 3:
#                     sound.stop()
#                     break
#                 else:
#                     if elapsed_time == wait_time:
#                         sound.play()
#                         elapsed_time = 0
#                         utils.colorBackgroundText(frame,  f'Drowsy', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
#                         break
#                     else:
#                         elapsed_time += time.time() - start_time
          
#             cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
#             cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)



#         end_time = time.time()-start_time
#         fps = frame_counter/end_time
#         print(fps)

#         frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
#         cv.imshow('frame', frame)
#         key = cv.waitKey(2)
#         if key==ord('q') or key ==ord('Q'):
#             break
#     cv.destroyAllWindows()
#     camera.release()

import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import pygame

pygame.init()
pygame.mixer.init()
sound = pygame.mixer.Sound("drowsy alarm.wav")

FONTS = cv.FONT_HERSHEY_COMPLEX
CLOSED_EYES_FRAME = 3
wait_time = 1

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

map_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture(0)

camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    return math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

def blinkRatio(landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    return (reRatio + leRatio) / 2

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    frame_counter = 0
    start_time = time.time()
    last_open_time = time.time()
    alarm_on = False

    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)

            if ratio < 3.5:
                last_open_time = time.time()
                if alarm_on:
                    sound.stop()
                    alarm_on = False
            else:
                if time.time() - last_open_time > wait_time:
                    if not alarm_on:
                        sound.play()
                        alarm_on = True
                    utils.colorBackgroundText(
                        frame, 'Drowsy', FONTS, 1.7,
                        (int(frame_height / 2), 100),
                        2, utils.YELLOW, pad_x=6, pad_y=6
                    )
                    cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                    cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                    cv.imwrite("Drowsy.jpg", frame)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

        elapsed = time.time() - start_time
        fps = frame_counter / elapsed
        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()
