from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from PIL import Image
from io import BytesIO

from django.views.generic import *
from webapp.forms import *

from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.models import LabelSet, ColumnDataSource

from sklearn.manifold import TSNE

from recommendations import *

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Treat these as global variables, do not want to reload them each time
embedding_matrix = get_embedding()
id_by_subreddit, subreddit_by_id = get_subreddit_dicts()
text_dict = load_subreddit_text()


class FrontPage(View):
    frontpage_template_name = "front_page.html"
    recommendation_template_name = "subreddit_recommendation.html"

    def get(self, request, *args, **kwargs):
        context = {'form': SimilarSubredditForm()}
        return render(request, self.frontpage_template_name, context)

    def post(self, request, *args, **kwargs):
        form = SimilarSubredditForm(request.POST)
        context = {'form': SimilarSubredditForm()}
        if not form.is_valid():
            return render(request, self.frontpage_template_name, context)
        subreddit = form.cleaned_data["subreddit"]

        # do real work to get recommendations
        if subreddit not in id_by_subreddit.keys():
            # Can't predict for unseen reddit
            messages.error(request, "This subreddit is not in the records.")
            return render(request, self.frontpage_template_name, context)

        recommended_subreddits, not_recommended = recommendations(subreddit, form.cleaned_data["num_similar"],
                                                 embedding_matrix, id_by_subreddit, subreddit_by_id)

        # list of recommended subreddits here
        context["subreddit"] = subreddit
        context["recommended_subreddits"] = recommended_subreddits
        context["not_recommended_subreddits"] = not_recommended

        # tsne plot here
        both_subreddits = recommended_subreddits + not_recommended
        script, div = tsnePlot(both_subreddits)
        context["tsne_script"] = script
        context["tsne_div"] = div

        # word cloud for the input subreddit here
        if subreddit in text_dict:
            # check to see if the dictionary has the subreddit
            context["wordcloud_present"] = True
        else:
            context["wordcloud_present"] = False

        return render(request, self.recommendation_template_name, context)


class WordCloudImageView(View):
    def get(self, request, *args, **kwargs):
        subreddit = kwargs["subreddit_name"]
        text = ''.join(text_dict[subreddit])

        # Generate a word cloud image
        wordcloud = WordCloud().generate(text)

        # Render the matplotlib image from wordcloud to a BytesIO buffer,
        # then return the contents of the buffer.
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        fig = plt.gcf()

        # Save it to a temporary buffer.
        buffer = BytesIO()
        fig.savefig(buffer, format="png")
        return HttpResponse(buffer.getvalue(), content_type="Image/png")


def tsnePlot(subreddit_names):
    # Returns two components, a script of the plot and the div element
    X = subreddit_embeddings(subreddit_names, embedding_matrix, id_by_subreddit)

    plot = figure(plot_width=400, plot_height=400)

    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)
    X_tsne = X_tsne[:len(subreddit_names), :]

    source = ColumnDataSource(data=dict(tsne_x= X_tsne[:, 0],
                                        tsne_y= X_tsne[:, 1],
                                        subreddit_names=subreddit_names))

    n = int(len(subreddit_names)/2)
    # add a circle renderer with a size, color, and alpha
    plot.circle(x=X_tsne[:n, 0], y=X_tsne[:n, 1], size=20, color="navy", alpha=0.5)
    plot.circle(x=X_tsne[n:, 0], y=X_tsne[n:, 1], size=20, color="red", alpha=0.5)

    labels = LabelSet(x='tsne_x', y='tsne_y', text='subreddit_names', level='glyph', x_offset=5, y_offset=5,
                      source=source)
    plot.add_layout(labels)

    return components(plot)

