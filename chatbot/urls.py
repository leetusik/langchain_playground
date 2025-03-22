from django.urls import path

from . import views

app_name = "chatbot"

urlpatterns = [
    # Web interface
    path("", views.chat_page, name="chat_page"),
    # API endpoints
    path("api/chat/", views.chat_api, name="chat_api"),
    path("api/chat/simple/", views.chat_simple, name="chat_simple"),
]
