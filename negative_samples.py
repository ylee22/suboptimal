from math import pow
from random import uniform, shuffle
import pandas as pd
from collections import defaultdict


class NegativeSampler:
    # This will sample the distribution provided to give negative examples. Following Mikolov et al support
    # scaling the weight of each subreddit by a power law.

    def __init__(self, graph_df, power=0.75):
        # Construct a probability for subreddit based on the number of edges
        # leading into that subreddit. Then apply a power transformation to
        # increase sampling towards the tail end of the distribution, and normalize
        # to produce a probability distribution.
        # dist = [(s1, sum(d.values())) for s1, d in graph_dict.items()]
        # df = pd.DataFrame(dist, columns=["subreddit", "total_edges"])

        # 1. change the input to a data frame, not a graph dictionary
        # 2. construct probability for subreddit based on the number of total edges
        # the subreddit is involved in (sum of count in source and dest), before it was calculated
        # by summing the total edge_count
        df = pd.DataFrame(graph_df.groupby('source_id').source_id.count()).join(graph_df.groupby('dest_id').dest_id.count())
        df.index.names = ['subreddit_id']
        df['unique_edge_counts'] = df[['source_id', 'dest_id']].sum(axis=1)
        df = df.reset_index()
        df["sample_weight"] = df.unique_edge_counts.apply(lambda n: pow(n, power))

        # In order to sample, construct the cumulative probability
        # distribution to search for the subreddit s_i, which has p_i,
        # by checking if a random variable in [0,1] lies in [c_i, c_i + p_i]
        # where c_i is the cumulative distribution of subreddits up to s_i in
        # this arbitrary but fixed ordering.
        df["prob_cutoff"] = df["sample_weight"].cumsum() / df["sample_weight"].sum()
        self.graph_df = graph_df
        self.sampling_distribution = [(r["prob_cutoff"], r["subreddit_id"]) for _, r in df.iterrows()]

        # Build a quick lookup for checking if there is a true positive
        self.fast_lookup = defaultdict(set)
        for _, row in graph_df.iterrows():
            self.fast_lookup[row["source_id"]].add(row["dest_id"])

    def sample(self, number_of_samples, check_if_positive=False):
        # Produce random edges between vertices based on the distribution over
        # subreddits which has been created. Normally in random negative sampling there is no
        # need to bother to check if the randomly selection is actually positive since
        # it's assumed that the odds of this are small, but is supported here.
        # If check_if_positive is set, may return less than the number of samples asked for.

        sample_subreddits, sample_edges = [], []
        # Need a sample for source and another for destination.
        uniform_rand = sorted([uniform(0, 1) for _ in range(2 * number_of_samples)])

        # Walk through both indices to see what goes in which bucket. Will keep stepping
        # forward with the uniform_rand_index and adding samples until the bucket boundary is crossed
        # and then will walk the sampling_dist_index forward until the current uniform_rand_index
        # is back inside of a bucket
        sampling_dist_index, uniform_rand_index = 0, 0

        while uniform_rand_index < len(uniform_rand) and sampling_dist_index < len(self.sampling_distribution):
            cum_p_value, subreddit_id = self.sampling_distribution[sampling_dist_index]
            if cum_p_value > uniform_rand[uniform_rand_index]:
                # fall into this bucket
                sample_subreddits.append(subreddit_id)
                uniform_rand_index += 1
            else:
                # Need to move forward to the next bucket
                sampling_dist_index += 1

        shuffle(sample_subreddits)
        # Generated twice as many as needed in order to fold the list into pairs.
        for s1, s2 in zip(sample_subreddits[0:number_of_samples], sample_subreddits[number_of_samples:]):
            if not check_if_positive:
                sample_edges.append((s1, s2))
            # 1. when checking if the edge exists in the data or not, use quick lookup
            # 2. check both source and destination
            if s2 not in self.fast_lookup[s1] and s1 not in self.fast_lookup[s2]:
                sample_edges.append((s1, s2))

        return sample_edges


## Everything after this is just to prove that this is actually
## sampling from the distribution which it it given. Don't need any of this
## once comfortable with what this is doing and that it works.
# graph_dict = {
#     "aww": {
#       "rarepuppers": 100,
#       "motocycles": 4,
#     },
#     "cats": {
#         "catshowerthoughts": 20
#     },
#     "askreddit": {
#         "treediffy": 200
#     }
# }
#
# neg = NegativeSampler(graph_dict, power=1)
# num_samples = 1000000
# samples = neg.new_sample(num_samples//2)
#
# real_dist = [(s1, sum(d.values())) for s1, d in graph_dict.items()]
# real_total = sum([v for k, v in real_dist])
# real_dist = {k: (100.0*v)/real_total for k, v in real_dist}
#
# from collections import defaultdict
# sample_dict = defaultdict(float)
# for s in samples:
#     sample_dict[s] += 1
# sample_keys = list(sample_dict.keys())
# sample_keys.sort(key=lambda s: sample_dict[s])
# sample_keys.reverse()
# for k in sample_keys[0:30]:
#     print(k, 100*sample_dict[k]/num_samples, real_dist[k])
