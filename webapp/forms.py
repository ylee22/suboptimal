
from django import forms

from webapp.models import *


class SimilarSubredditForm(forms.Form):
    subreddit = forms.CharField(max_length=40, label="Subreddit")
    num_similar = forms.IntegerField(initial=10, min_value=1, required=True, label="Number of recommendations")

