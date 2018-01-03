from django.conf.urls import url
from django.contrib import admin
import handgesturerecognition.views

urlpatterns = [
		url(r'^',handgesturerecognition.views.predict),
        url(r'^admin/', admin.site.urls),
        url(r'^handgesturerecognition/predict/$', handgesturerecognition.views.predict),
]


