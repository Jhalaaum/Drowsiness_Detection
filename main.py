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
