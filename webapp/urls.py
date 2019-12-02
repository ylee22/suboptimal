from django.conf.urls import url
from django.urls import path

from webapp.views import *

app_name = "webapp"
urlpatterns = [
    path(r'', FrontPage.as_view(), name='front_page'),
    path('wordcloud/<str:subreddit_name>/', WordCloudImageView.as_view(), name='word_cloud'),

]

