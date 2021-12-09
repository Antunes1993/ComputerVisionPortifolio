from django.urls import path
from .views import views

urlpatterns = [
    path('inicio', views.ferramenta01, name='inicio'),
    path('face_detector', views.face_detector_page, name='face_detector'),
    path('facial_mask_detector', views.facial_mask_detector_page, name='facial_mask_detector'),
    path('atention_detector', views.atention_detector_page, name='atention_detector'),

    path('video_feed', views.video_feed, name='video_feed'),
    path('video_feed_atention_detector', views.video_feed_atention_detector, name='video_feed_atention_detector'),
    path('video_feed_facial_mask_detector', views.video_feed_facial_mask_detector, name='video_feed_facial_mask_detector'),

    path('webcam_feed', views.webcam_feed, name='webcam_feed'),


]