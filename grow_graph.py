from datetime import date
from typing import List, Dict, Tuple, Set
from itertools import product
from time import time
from collections import Counter
from random import shuffle
from prawcore.exceptions import Forbidden, NotFound
from collections import defaultdict
import re
import json
import gzip
import pickle
import os
import argparse
import glob
import sys

from praw import Reddit
from praw.models import Submission, Redditor, Subreddit


class GrowGraph:
    reddit = Reddit(client_id='eOdy1q-rTCofCQ', client_secret='XBiBKRfXqCpIJezodNyxnrB8EXI', user_agent='suboptimal')

    def __init__(self, starting_points=[reddit.subreddit('all')], subreddit_link_counts={}, visited_users=set(),
                 waitlist_users=[], max_submissions=100, visited_subreddits=set(), max_subreddits=300,
                 max_user_posts=20, max_users=1000, users_403=set(), error_counts=defaultdict(int)):
        # If starting the first time and don't have any starting points, use /r/all to get started
        # starting_points is a list of subreddits
        self.start = starting_points  # list of subreddit objects

        # subreddit dict of dicts with other subreddits. These are stored with id numbers instead of subreddit
        # names since the dictionary is huge.
        self.subred_link_counts = subreddit_link_counts
        # Keep track of the subreddit numbers
        self.subreddit_by_id = {}
        self.next_subreddit_id = 1

        self.visited_users = visited_users  # set of user name strings
        self.new_users = waitlist_users  # list of redditor objects
        self.max_submissions = max_submissions  # int of maximum submissions for a given subreddit
        self.visited_subred = visited_subreddits  # set of subreddit name strings
        self.max_subred = max_subreddits  # int of maximum subreddits to go through each function call
        self.max_user_posts = max_user_posts  # int of maximum subreddits for a given user history
        self.max_users = max_users  # int of maximum users to go through each function call
        self.error_users = users_403  # set of user name strings that caused 403 forbidden request error
        self.error_counts = error_counts  # dictionary of error names and the number of times this occurred

        self.subreddit_descriptions = defaultdict(str)  # the description text for each subreddit
        self.iteration_number = 0  # the number of iterations taken
        self.maximum_redditer_edge_strength = 9  # the maximum amount to add to a given edge from just one redditer

    def grab_users(self):
        # Step 1. Grab new users in order to go through their history
        newly_discovered_users = set()

        # shuffle subreddit list before going through it
        shuffle(self.start)

        # only go through a max number of subreddits for each round
        for subreddit in self.start[:self.max_subred]:
            # if a subreddit hasn't been already scraped
            if subreddit.display_name not in self.visited_subred:
                try:
                    # add subreddit to the set of visited subreddits (Set)
                    self.visited_subred.add(subreddit.display_name)
                    # grab all the authors of the subreddit submissions
                    for submission in subreddit.new(limit=self.max_submissions):
                        if submission.author is not None:
                            newly_discovered_users.add(submission.author)

                    # Record the description text too
                    #self.subreddit_descriptions[subreddit.display_name] = self.get_subreddit_description(subreddit)
                except Forbidden:
                    print("Forbidden 403 exception, skipping")

        # update the list of starting subreddits, remove the first self.max_subred number of items
        self.start = self.start[self.max_subred:]

        # Step 2. Filter for new users that haven't already encountered
        # filter newly discovered users for users already went through
        # returns a set of Redditor objects and the string of the new names (to add to the known_user_names)
        self.filter_known_users(newly_discovered_users)

    def grab_user_submissions(self):
        # Collect user post histories
        newly_discovered_subreddits = set()

        # shuffle user list before going through it
        shuffle(self.new_users)

        for redditor in self.new_users[:self.max_users]:
            # Record what else this person did. Limit to the last year.
            submissions = self.recent_redditor_submissions(redditor)
            # this is a list of subreddits posted for the given redditor
            user_subreddits = [s.subreddit for s in submissions]

            # Do not want any u_myusername per use "subreddits", to filter them out
            user_subreddits = [s for s in user_subreddits if not s.display_name.startswith("u_")]

            # add new subreddits to be explored next time the function is called
            # use set because users can submit to the same subreddit multiple times (there are duplicates in list)
            newly_discovered_subreddits = newly_discovered_subreddits.union(user_subreddits)

            # Add counts for the number of links between related subreddits based on user post history
            # Todo: s.name is probably more technically correct, the display_name may be ambiguous. But this is easier
            # for debugging purposes at least.
            user_subreddit_names = [s.display_name for s in user_subreddits]

            # store the subreddit description for a given display_name
            # todo store just the top part of the subreddit description
            # This required parsing the public description since it has lots of
            # markdown and stuff that I really don't care about.
            # for s in user_subreddits:
            #     self.subreddit_descriptions[s.display_name] = s.public_description
            #     print(s.description)

            # there are duplicates in the user_subreddit_names
            subreddit_counts = Counter(user_subreddit_names)

            # iterate through every permutation of the list of subreddits for a given user's post
            for sub1, sub2 in product(subreddit_counts.keys(), subreddit_counts.keys()):
                sub1_id, sub2_id = self.get_subreddit_id(sub1), self.get_subreddit_id(sub2)
                # Cartesian product
                if sub1 == sub2:
                    # Don't link from a subreddit to itself
                    # or skip when this subreddit has been explored to death already
                    continue
                if sub1_id not in self.subred_link_counts.keys():
                    # initialize new dict if this sub1 has not been seen before
                    self.subred_link_counts[sub1_id] = {}
                if sub2 not in self.subred_link_counts[sub1_id].keys():
                    # initialize new dict for sub2 if this is a new link (sub1, sub2)
                    self.subred_link_counts[sub1_id][sub2_id] = 0

                # add link between sub1 and sub2, but don't allow just one person to fully link two subreddits
                edge_contribution = min(self.maximum_redditer_edge_strength, subreddit_counts[sub1] * subreddit_counts[sub2])
                self.subred_link_counts[sub1_id][sub2_id] += edge_contribution

        self.start.extend(newly_discovered_subreddits)
        # update the list of starting subreddits
        self.new_users = self.new_users[self.max_users:]

    def get_subreddit_id(self, display_name: str) -> int:
        # Get the id of a named subreddit. If this subreedit has never been seen before, assign an index and return it.
        if display_name not in self.subreddit_by_id.keys():
            self.subreddit_by_id[display_name] = self.next_subreddit_id
            self.next_subreddit_id += 1
        return self.subreddit_by_id[display_name]

    def get_subreddit_description(self, subreddit: Subreddit) -> str:
        # Get the description of a subreddit, and take just the short description text at the top of it
        # The markup separates sections of the description with ###
        try:
            # In case the subreddit text is not cut correctly limit to 300 characters.
            fragments = re.split('([#*])', subreddit.description)
            filtered_fragments = [s for s in fragments if len(s.strip(" ")) > 6]
            if len(filtered_fragments) > 0:
                return filtered_fragments[0]
            else:
                return ""
        except NotFound:
            return ""

    def is_submission_within_year(self, submission: Submission) -> bool:
        # Check if a submission was created within the last year
        df = date.fromtimestamp(submission.created_utc) - date.today()
        return abs(df.days) <= 365

    def recent_redditor_submissions(self, redditor: Redditor) -> List[Submission]:
        submissions = []
        try:
            for s in redditor.submissions.new(limit=self.max_user_posts):
                if self.is_submission_within_year(s):
                    submissions.append(s)
        except Forbidden as e:
            # collect exceptions
            # <class 'prawcore.exceptions.Forbidden'>
            # received 403 HTTP response
            self.error_users.add(redditor.name)
            self.error_counts[e.__class__.__name__] += 1
        except Exception as e:
            self.error_counts[e.__class__.__name__] += 1
        return submissions

    def filter_known_users(self, new_users: Set[Redditor]):
        # takes a set of newly found users, checks to see if their history has already been scraped
        # if they're new, add them to the list of new users
        for user in new_users:
            if user.name not in self.visited_users:
                # if this user hasn't been encountered before, add to the list of new users
                self.new_users.append(user)
                # add newly discovered users to the set of known users
                # so that I don't go through their post history again
                self.visited_users.add(user.name)



# The outside function, grow_graph, will be responsible for calling walk_to_new_subreddits,
# holding onto it's return values, saving things to disk, printing status and stats after each step
# (like how many subreddits have been explored, how many new), and then putting the newly discovered subreddits
# into the starting_points variable for when it gets called again
# Initially starting points will just be []
def go_for_a_walk(reload_from_pickle: bool = False):
    g = GrowGraph()
    start_program = time()

    if reload_from_pickle:
        pickles = sorted(glob.glob('subreddit_graph_*.p'))
        if len(pickles) == 0:
            print("Could not find any pickles to load, quitting")
            sys.exit(1)
        filename = pickles[-1]
        print("Reloading from {}".format(filename))
        with open(filename, "rb") as f:
            g = pickle.load(f)
            print("Reloaded successfully on iteration: {}".format(g.iteration_number))

    while True:
        print("Starting iteration")
        start_time = time()
        # class variables just get updated internally, shouldn't have to pass anything

        # Building up too many users in the queue is the main reason for out of memory crashing
        if len(g.new_users) > 5000:
            print("More than 5000 new_users, going to wait to work through the backlog")
        else:
            g.grab_users()

        g.grab_user_submissions()
        print('finished walk: {}'.format(g.iteration_number))
        print('scraped users: {}'.format(len(g.visited_users)))
        print('user waitlist: {}'.format(len(g.new_users)))
        print('total visited subreddits: {}'.format(len(g.visited_subred)))
        print('starting subreddits {}'.format(len(g.start)))
        print('number of subreddits in dictionary {}'.format(len(g.subred_link_counts)))

        # After every 5th time the dictionary is expanded, write a pickle of the entire class which are being
        # used, along with the graph edges in json. The reason for the pickles are to make it quick
        # to restart the scrapper process, whereas the json is the finished output which will be used
        # in other parts of the program.
        if g.iteration_number % 5 == 0:
            # Dump a new pickle
            pickle.dump(g, open('subreddit_graph_{}.p'.format(str(g.iteration_number).zfill(4)), 'wb'))

            with gzip.open('current_subreddit_graph.json.gz', 'wt') as outfile:
                json.dump(g.subred_link_counts, outfile)
            with gzip.open('current_subreddit_ids.gz', 'wt') as outfile:
                json.dump(g.subreddit_by_id, outfile)
            with gzip.open('current_subreddit_descriptions.json.gz', 'wt') as outfile:
                json.dump(g.subreddit_descriptions, outfile)

            # Don't keep too many pickles lying around, using up disk space
            pickles = sorted(glob.glob('subreddit_graph_*.p'))
            if len(pickles) > 5:
                os.remove(pickles[0])

        # Format how long this took in a legible fashion
        dt = time() - start_time
        hours = int(dt) // 3600
        minutes = int(dt - 3600 * hours) // 60
        sec = int(dt - 3600 * hours - 60 * minutes)
        print("Finished call to walk_to_new_subreddits, time elapsed: {}:{}:{} sec\n".format(hours, minutes, sec))

        # Format how long this took in a legible fashion
        dt = time() - start_program
        hours = int(dt) // 3600
        minutes = int(dt - 3600 * hours) // 60
        sec = int(dt - 3600 * hours - 60 * minutes)
        print("Time elapsed since starting: {}:{}:{} sec\n".format(hours, minutes, sec))

        g.iteration_number += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restart', action="store_true")
    args = parser.parse_args()

    # example of how to grow the graph
    go_for_a_walk(args.restart)
