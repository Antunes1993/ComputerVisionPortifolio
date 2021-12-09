import queue
import datetime
import threading
import numpy as np
from imutils import resize
import cv2, os, urllib.request
from django.conf import settings


class FaceDetector(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.initial_frontal_face_detected = False
        self.initial_left_profile_detected = False
        self.initial_right_profile_detected = False

    def __del__(self):
        self.video.release()

    def get_frame(self):
        os.chdir("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\Miscelanias\\ArquivosReferencia")
        # Instancia os classificadores em cascata com os haar-cascades
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        ret, frame = self.video.read()
        #frame, face_detected = self.faces_detection(frame)

        frame, face_detected, profile_left_face_detected, profile_right_face_detected = self.faces_detection(frame)
        #print ("Fade detected: ", face_detected)
        #print("Profie Face left detected: ", profile_left_face_detected)
        #print("Profie Face right detected: ", profile_right_face_detected)

        frame = self.faces_monitor(frame, face_detected, profile_left_face_detected, profile_right_face_detected)
        self.show_current_date(frame)

        ret2, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def show_current_date(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 0)
        thickness = 1
        org = (450, 20)
        now = datetime.datetime.now()
        frame = cv2.putText(frame, now.strftime("%d-%m-%Y %H:%M:%S"), org, font, fontScale, color, thickness, cv2.LINE_AA)

    def face_classification(self, classifier, frame):
        faces = classifier.detectMultiScale(frame, 1.1, 5)
        return faces


    def faces_detection(self,frame):
        os.chdir("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\Miscelanias\\ArquivosReferencia")
        # Instancia os classificadores em cascata com os haar-cascades
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        profile_face_cascade = cv2.CascadeClassifier('haarcascade_profile.xml')

        que1 = queue.Queue()
        que2 = queue.Queue()
        que3 = queue.Queue()

        frontal_face_detected = False
        profile_face_detected = False

        # Converte para espaço de cores YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        hist_equalization_result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


        # Detecta faces com histograma equalizado
        #t1 = threading.Thread(target=lambda q, arg1: q.put(self.face_classification(faces)), args=(que1, face_cascade, frame))
        #t2 = threading.Thread(target=lambda q, arg1: q.put(self.face_classification(faces)), args=(que2, profile_face_cascade, frame))
        #t3 = threading.Thread(target=lambda q, arg1: q.put(self.face_classification(faces)), args=(que3, profile_face_cascade, cv2.flip(frame,1)))

        #t1.start()
        #t2.start()
        #t3.start()

        #t1.join()
        #t2.join()
        #t3.join()

        #faces = que1.get()
        #faces_profile_left = que2.get()
        #faces_profile_right = que3.get()

        faces = self.face_classification(face_cascade, frame) # hist_equalization_detection
        faces_profile_left = self.face_classification(profile_face_cascade, frame) # hist_equalization_detection
        faces_profile_right = self.face_classification(profile_face_cascade, cv2.flip(frame,1))  # hist_equalization_detection


        # Detecção de faces
        for (x, y, w, h) in faces:
            #cv2.rectangle(hist_equalization_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #roi_color = frame[y:y + h, x:x + w]
            self.initial_frontal_face_detected = True

        # Detecção de faces
        for (x, y, w, h) in faces_profile_left:
            #cv2.rectangle(hist_equalization_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #roi_color = frame[y:y + h, x:x + w]
            profile_face_detected = True
            self.initial_left_profile_detected = True

        for (x, y, w, h) in faces_profile_right:
            #cv2.rectangle(hist_equalization_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #roi_color = frame[y:y + h, x:x + w]
            self.initial_right_profile_detected = True

        return frame, self.initial_frontal_face_detected, self.initial_left_profile_detected, self.initial_right_profile_detected

    def faces_monitor(self,frame, frontal_face_detect_status, left_face_detect_status, right_face_detect_status):

        #Nenhum rosto detectado
        if (frontal_face_detect_status == False and left_face_detect_status == False and right_face_detect_status == False):
            imgFace = cv2.imread("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame01.png")

        #Detecções individuais
        elif(frontal_face_detect_status == True and left_face_detect_status == False and right_face_detect_status == False):
            imgFace = cv2.imread("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame05.png")

        elif (frontal_face_detect_status == False and left_face_detect_status == True and right_face_detect_status == False):
            imgFace = cv2.imread("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame03.png")

        elif (frontal_face_detect_status == False and left_face_detect_status == False and right_face_detect_status == True):
            imgFace = cv2.imread("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame04.png")

        #Detecções compartilhadas
        elif(frontal_face_detect_status == True and left_face_detect_status == True and right_face_detect_status == False):
            imgFace = cv2.imread("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame06.png")

        elif (frontal_face_detect_status == True and left_face_detect_status == False and right_face_detect_status == True):
            imgFace = cv2.imread("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame07.png")

        elif (frontal_face_detect_status == False and left_face_detect_status == True and right_face_detect_status == True):
            imgFace = cv2.imread("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame08.png")

        elif (frontal_face_detect_status == True and left_face_detect_status == True and right_face_detect_status == True):
            imgFace = cv2.imread("C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame02.png")


        imgFace = resize(imgFace, width=250, height=250)
        imgFace = cv2.cvtColor(imgFace, cv2.COLOR_BGR2RGB)

        frame_h, frame_w, frame_c = frame.shape
        overlay = np.zeros((frame_h, frame_w, frame_c), dtype=np.uint8)
        imgFace_h, imgFace_w, imgFace_c = imgFace.shape

        for i in range(0, imgFace_h):
            for j in range(0, imgFace_w):
                if imgFace[i, j][2] != 0:
                    overlay[i + 10, j + 10] = imgFace[i, j]

        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        return frame


class IPWebCam(object):
    def __init__(self):
        self.url = "http://192.168.0.134:8080/video"

    def __del__(self):
        cv2.destroyAllWindows()

    def get_frame(self):
        imgRest = urllib.request.urlopen(self.url)
        imgNp = np.array(bytearray(imgRest.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp,1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(img, (640,480), interpolation=cv2.INTER_LINEAR)
        frame_flip = cv2.flip(resize,1)
        ret, jpeg = cv2.imencode('.jpg', frame_flip)
        return jpeg.tobytes()