from prawcore.exceptions import Forbidden, NotFound
from collections import defaultdict
import re
import json
import gzip
import pickle
import os
import argparse
from time import time
import sys

from praw import Reddit
from praw.models import Submission, Redditor, Subreddit


def expand_texts(reddit, texts, subreddit_list, num_subreddits):
    print("Processed texts: {}, subreddits to go: {}k".format(len(texts), len(subreddit_list)//1000))
    for subreddit_name in subreddit_list[:num_subreddits]:
        # print("Getting posts for {}".format(subreddit_name))
        subreddit = reddit.subreddit(subreddit_name)
        post_titles = []
        try:
            for submission in subreddit.hot(limit=50):
                try:
                    post_titles.append(submission.title)
                except Forbidden:
                    print("Skipping post due to forbidden exception")
                except NotFound:
                    print("Skipping post due to 404 exception")
                except Exception as e:
                    print("Skipping due to other: {}".format(e))
            texts[subreddit_name] = post_titles
        except Exception as e:
            print("Could not get {}.hot because: {}".format(subreddit_name, e))
    return texts, subreddit_list[num_subreddits:]


if __name__ == '__main__':
    ids_file = "current_subreddit_ids.gz"
    graph_file = "current_subreddit_graph.json.gz"

    reddit = Reddit(client_id='eOdy1q-rTCofCQ', client_secret='XBiBKRfXqCpIJezodNyxnrB8EXI', user_agent='suboptimal')

    print("Reading id file")
    with gzip.open(ids_file, "rb") as f:
        id_by_subreddit = json.load(f)
    print("Sample keys:")
    print(list(id_by_subreddit.keys())[0:10])
    print("\nLoading graph_dict")
    with gzip.open(graph_file, "rb") as f:
        graph_dict = json.load(f)
    print("Sample keys:")
    print(list(graph_dict.keys())[0:10])

    subreddit_by_id = {int(v): k for k, v in id_by_subreddit.items()}
    print("\nLoaded graph_dict with {} keys".format(len(graph_dict.keys())))

    def count_keys(d):
        return sum([int(n) for n in d.values()])
    link_counts = [(subreddit_by_id[int(k)], count_keys(v)) for k, v in graph_dict.items()]
    link_counts.sort(key=lambda t: t[1], reverse=True)

    # free up memory since the graph dict can be huge
    del graph_dict
    print(link_counts[0:10])

    # Restore where we left off with texts
    if os.path.exists("subreddit_texts.json.gz"):
        with gzip.open("subreddit_texts.json.gz", "rb") as f:
            texts = json.load(f)
            print("Reloaded partial progress successfully")
    else:
        texts = {}

    finished_subreddits = set(texts.keys())
    subreddit_list = [s for s, _ in link_counts if s not in finished_subreddits]

    start_program = time()
    iteration_count = 0
    while len(subreddit_list) > 0:
        iteration_count += 1
        texts, subreddit_list = expand_texts(reddit, texts, subreddit_list, 200)

        # Format how long this took in a legible fashion
        dt = time() - start_program
        hours = int(dt) // 3600
        minutes = int(dt - 3600 * hours) // 60
        sec = int(dt - 3600 * hours - 60 * minutes)
        print("Time elapsed since starting: {}:{}:{} sec\n".format(hours, minutes, sec))

        if iteration_count % 10 == 0:
            # Save
            print("Saving results so far")
            with gzip.open('subreddit_texts.json.gz', 'wt') as outfile:
                json.dump(texts, outfile)

    print("Finished all data")
