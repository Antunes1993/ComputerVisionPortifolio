import cv2, os, urllib.request
import numpy as np
from django.conf import settings
import os
import numpy as np
import datetime
from imutils import resize
import dlib
from numpy.core.einsumfunc import _einsum_dispatcher
from scipy.spatial import distance as dist

class AtentionDetector(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.initial_frontal_face_detected = False
        self.initial_left_profile_detected = False
        self.initial_right_profile_detected = False

        # Fontes de texto
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.7
        self.color = (0, 0, 255)
        self.thickness = 2

        # Posicionamento de mensagens
        self.org = (450, 20)
        self.org2 = (20, 20)
        self.org3 = (30, 450)

        # Classificador
        self.classificador_dlib_68 = "C:\\Users\\feoxp7\\Desktop\\ESTUDO\\detector_de_sono\\classificadores\\shape_predictor_68_face_landmarks.dat"
        self.classificador_dlib = dlib.shape_predictor(self.classificador_dlib_68)
        self.detector_face = dlib.get_frontal_face_detector()

        # Pontos fiduciais
        self.FACE = list(range(17, 68))
        self.FACE_COMPLETA = list(range(0, 68))
        self.LABIO = list(range(48, 61))
        self.SOMBRANCELHA_DIREITA = list(range(17, 22))
        self.SOMBRANCELHA_ESQUERDA = list(range(22, 27))
        self.OLHO_DIREITO = list(range(36, 42))
        self.OLHO_ESQUERDO = list(range(42, 48))
        self.NARIZ = list(range(27, 35))
        self.MANDIBULA = list(range(0, 17))

        # Variaveis de controle
        self.modo_apresentacao = True
        self.marcos_obtidos = []

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        # Detector de Faces
        retangulos = self.detector_face(frame, 1)

        if (len(retangulos)) > 0:
            frame = self.detector_faces(frame, retangulos, self.modo_apresentacao)

        #Pontos fiduciais
        if (len(retangulos)) > 0:
            frame, self.marcos_obtidos = self.marcos_faciais(frame, retangulos, self.modo_apresentacao)

        # Casca convexa
        if (len(retangulos)) > 0 and (len(self.marcos_obtidos)) > 0:
            frame = self.anotar_marcos_casca_convexa(frame, self.marcos_obtidos, retangulos, self.modo_apresentacao)

        if (len(self.marcos_obtidos)) > 0:
            valor_olho_esquerdo = self.aspecto_razao_olhos(self.marcos_obtidos[0][self.OLHO_ESQUERDO])
            valor_olho_direito = self.aspecto_razao_olhos(self.marcos_obtidos[0][self.OLHO_DIREITO])
            valor_labios = self.aspecto_razao_boca(self.marcos_obtidos[0][self.LABIO])

            #print("Olho esquerdo: ", valor_olho_esquerdo)
            #print("Olho direito: ", valor_olho_direito)
            #print("Labios: ", valor_labios)
            #frame = cv2.putText(frame, str(valor_labios), org2, font, fontScale, color, thickness, cv2.LINE_AA)

            if valor_labios > 0.5 or valor_olho_esquerdo < 0.25 or valor_olho_direito < 0.25:
                frame = cv2.putText(frame, "Detectado reducao de atencao causada por cansaco", self.org3, self.font, self.fontScale,self.color, self.thickness, cv2.LINE_AA)


        #frame = self.faces_monitor
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


    def detector_faces(self, frame, retangulos, modo_apresentacao):
        if (modo_apresentacao == True):
            for k, d in enumerate(retangulos):
                cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        return frame


    def marcos_faciais(self, frame, retangulos, modo_apresentacao):
        marcos = []
        if len(retangulos) > 0:
            for ret in retangulos:
                marcos.append(np.matrix([[p.x, p.y] for p in self.classificador_dlib(frame, ret).parts()]))

        if len(marcos) > 0:
            for marco in marcos:
                for idx, ponto in enumerate(marco):
                    centro = (ponto[0, 0], ponto[0, 1])
                    if (modo_apresentacao == True):
                        cv2.circle(frame, centro, 2, (45, 255, 255), -1)
                    # cv2.putText(frame, str(idx), (centro[0] - 10, centro[1] - 10), font, fontScale, color, thickness, cv2.LINE_AA)

        return frame, marcos


    def anotar_marcos_casca_convexa(self, frame, marcos, retangulos, modo_apresentacao):
        if (len(retangulos)) == 0:
            return None

        for idx, ret in enumerate(retangulos):
            marco = marcos[idx]
            pontos_olho_esquerdo = cv2.convexHull(marco[self.OLHO_ESQUERDO])

            pontos_olho_direito = cv2.convexHull(marco[self.OLHO_DIREITO])

            pontos_labio = cv2.convexHull(marco[self.LABIO])

            if (self.modo_apresentacao == True):
                cv2.drawContours(frame, [pontos_olho_esquerdo], -1, (0, 255, 0), 2)
                cv2.drawContours(frame, [pontos_olho_direito], -1, (0, 255, 0), 2)
                cv2.drawContours(frame, [pontos_labio], -1, (0, 255, 0), 2)

        return frame

    def aspecto_razao_olhos(self, pontos_olhos):
        a = dist.euclidean(pontos_olhos[1], pontos_olhos[5])
        b = dist.euclidean(pontos_olhos[2], pontos_olhos[4])
        c = dist.euclidean(pontos_olhos[0], pontos_olhos[3])

        aspecto_razao = (a + b) / (2.0 * c)
        return aspecto_razao

    def aspecto_razao_boca(self, pontos_boca):
        a = dist.euclidean(pontos_boca[3], pontos_boca[9])
        b = dist.euclidean(pontos_boca[2], pontos_boca[10])
        c = dist.euclidean(pontos_boca[4], pontos_boca[8])
        d = dist.euclidean(pontos_boca[0], pontos_boca[6])

        aspecto_razao = (a + b + c) / (3.0 * d)

        return aspecto_razao

    def faces_monitor(self,frame):
        imgFace = cv2.imread(
            "C:\\Users\\feoxp7\\Desktop\\ESTUDO\\ComputerVisionPortifolio\\ComputerVisionPortifolio\\App01\\static\\App01\\img\\frame01.png")


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