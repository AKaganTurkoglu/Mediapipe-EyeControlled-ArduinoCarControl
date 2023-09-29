import cv2 as cv 
import numpy as np
import mediapipe as mp 
import time
import serial


ser = serial.Serial("COM4", 9600, timeout = 1) #Change your port name COM... and your baudrate _

def retrieveData():
    ser.write(b'1')
def sifir():
    ser.write(b'0')    
def iki():
    ser.write(b'2')
def uc():
    ser.write(b'3')
def dort():
    ser.write(b'4')
def bes():
    ser.write(b'5')

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
RIGHT_IRIS = [474,475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # left eye left most landmark
L_H_RIGHT = [133]  # left eye right most landmark
R_H_LEFT = [362]  # right eye left most landmark
R_H_RIGHT = [263]  # right eye right most landmark
R_V_UP = [386]  # right eye up most landmark
R_V_DOWN = [374]  # right eye down most landmark

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def horizontal_iris_position(iris_center,left_point,right_point):
    center_to_right_distance = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(left_point, right_point)
    ratio = center_to_right_distance / total_distance
    return ratio

def vertical_iris_position(iris_center,up_point,down_point):
    center_to_up_distance = euclidean_distance(iris_center, up_point)
    center_to_down_distance = euclidean_distance(iris_center, down_point)
    total_distance = euclidean_distance(up_point, down_point)
   
    up_ratio = center_to_up_distance / total_distance
    down_ratio = center_to_down_distance / total_distance
    print("center_to_up_distance",center_to_up_distance)
    print("total_distance",total_distance)
    print("up_ratio",up_ratio)
    if total_distance < 9.5:
        return "closed"
    elif total_distance <= 25 or up_ratio > .51:
        return "down"
    elif up_ratio <= .21 or total_distance >= 43:
        return "up"
    elif .21 < up_ratio <= .51:
        return "center"


def eye_direction(iris_center, left_point, right_point, up_point, down_point):
    hor_iris_ratio = horizontal_iris_position(iris_center, left_point, right_point)
    ver_iris_ratio = vertical_iris_position(iris_center[1], up_point[1], down_point[1])
 #   print("ver_iris_ratio",ver_iris_ratio)
    if hor_iris_ratio > 0.42 and hor_iris_ratio <= 0.59: # horizontal center
       return ver_iris_ratio
    elif hor_iris_ratio > 0.59: # horizontal left
        return 'left'
    elif hor_iris_ratio <= 0.42: # horizontal right
        return 'right'
    

def get_direction_to_int(direction):
    if direction == 'left':
        return 2
    elif direction == 'right':
        return 4
    elif direction == 'up':
        return 1
    elif direction == 'down':
        return 3
    elif direction == 'center':
        return 0
    elif direction == 'closed':
        return 5

def preprocess():
    global dizi
    max = -1
    j = 0
    k =0
    for i in dizi:
        if i > max:
            max = i
            k = j
        dizi[j] = 0
        j+=1
    return k


dizi = [0] * 6
dizi[0] = 0
dizi[1] = 0
dizi[2] = 0
dizi[3] = 0
dizi[4] = 0
dizi[5] = 0
counter=-20
a=0
bitis = 999
cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    BASLA = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255,255,255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[R_H_LEFT][0], 3, (0,255,255), -1, cv.LINE_AA)
            iris_direction = eye_direction(center_right, mesh_points[R_H_LEFT][0], mesh_points[R_H_RIGHT][0], mesh_points[R_V_UP][0], mesh_points[R_V_DOWN][0])
            counter+=1
        #    print(counter)
            yyy = get_direction_to_int(iris_direction)
            if yyy == None:
                indis = 0
            else:
                indis = int(yyy)

            sonuc = bitis - BASLA
            dizi[indis] = dizi[indis] + 1
            if sonuc > 0.95:
                BASLA = time.time()
                counter = 0
                uInput = preprocess() 
                #get_direction_to_int(iris_direction)
                if uInput == 1:
                    retrieveData()
                    print(uInput)
                elif uInput == 2:
                    iki()
                    print(uInput)
                elif uInput == 3:
                    uc()
                    print(uInput)
                elif uInput == 4:
                    dort()
                    print(uInput)
                elif uInput == 5:
                    bes()      
                    print(uInput)
                else:
                    sifir()  
                    print("0")   
        #    print(iris_direction)
        #    print(dizi)
            cv.putText(frame, iris_direction, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow('img', frame)
        key = cv.waitKey(10)
        bitis = time.time()
        sonuc = bitis - BASLA
    #    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",sonuc)
        if key ==ord('q'):
            break
#s.close()
cap.release()
cv.destroyAllWindows()

