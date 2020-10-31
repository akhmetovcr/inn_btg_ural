from django.urls import path
from .import views


urlpatterns = [
    path('test', views.post_index, name='post_index'),
    path('test/', views.post_index, name='post_index'),
]
