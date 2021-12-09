from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from ..camera import VideoCamera, IPWebCam
from ..computer_vision_algorithms.face_detector import FaceDetector
from ..computer_vision_algorithms.atention_detector import AtentionDetector
from ..computer_vision_algorithms.facial_mask_detector import FacialMaskDetector

from django.http import HttpResponse
import pathlib

# Create your views here.
def ferramenta01(request):
    return render(request, '../templates/index.html');

def face_detector_page(request):
    return render(request, '../templates/face_detector.html');

def facial_mask_detector_page(request):
    return render(request, '../templates/covid_mask_detection.html');

def atention_detector_page(request):
    return render(request, '../templates/atention_detector.html');


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    response = StreamingHttpResponse(gen(FaceDetector()), content_type='multipart/x-mixed-replace; boundary=frame')
    return response

def video_feed_atention_detector(request):
    response = StreamingHttpResponse(gen(AtentionDetector()), content_type='multipart/x-mixed-replace; boundary=frame')
    return response

def video_feed_facial_mask_detector(request):
    response = StreamingHttpResponse(gen(FacialMaskDetector()), content_type='multipart/x-mixed-replace; boundary=frame')
    return response

def webcam_feed(request):
    return StreamingHttpResponse(gen(IPWebCam()), content_type='multipart/x-mixed-replace; boundary=frame')
