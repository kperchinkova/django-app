from django.conf.urls import url
from django.contrib import admin
import face_detector.views

urlpatterns = [
		url(r'^',face_detector.views.predict),
        url(r'^admin/', admin.site.urls),
        url(r'^face_detection/predict/$', face_detector.views.predict),
]


