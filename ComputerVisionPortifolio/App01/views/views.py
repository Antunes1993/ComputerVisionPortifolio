from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from ..camera import VideoCamera, IPWebCam
from django.http import HttpResponse
import pathlib



# Create your views here.
def ferramenta01(request):
    return render(request, '../templates/index.html');

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    response = StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')
    return response

def webcam_feed(request):
    return StreamingHttpResponse(gen(IPWebCam()), content_type='multipart/x-mixed-replace; boundary=frame')

def mensagem_feed(request):
    response = StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=gray')
    return response

def mensagem_teste(request):
    mensagem = "223"
    return HttpResponse(request.POST['text'])