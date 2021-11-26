from django.urls import path
from .views import views

urlpatterns = [
    path('ferramenta01', views.ferramenta01, name='ferramenta_01'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('webcam_feed', views.webcam_feed, name='webcam_feed'),
    path('mensagem_feed', views.mensagem_feed, name='mensagem_feed'),
    path('mensagem_teste', views.mensagem_teste, name='mensagem_teste'),

]