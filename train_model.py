import tensorflow as tf
import numpy as np
import pandas as pd
import json
import gzip
from time import time
from keras import backend
from keras.layers import Input, Embedding, dot, Reshape, Activation
from keras.models import Model
import preprocess_graph
from negative_samples import NegativeSampler


def auc(y_true, y_predictions):
    auc_metric = tf.metrics.auc(y_true, y_predictions)[1]
    backend.get_session().run(tf.local_variables_initializer())
    return auc_metric


def attainable_symmetric_precision_at_k(model, training_df, subreddit_by_id, k):
    # For each source subreddit, find the k most likely subreddits predicted to be true. Of those, compute
    # the fraction which are true positives. This gives a precision per subreddit at k. The mean of all
    # subreddit precision at k is returned at the end.
    #
    # This is different from a normal precision at k because for each subreddit it only computes the number of
    # true examples in the top min(k, num_true_positives) so for k > num_true_positives the metric does not go down.
    predictions = model.predict([np.array(training_df.source_id), np.array(training_df.dest_id)])

    src_data = [(p[0], subreddit_by_id[src], subreddit_by_id[dst], label) for p, src, dst, label in
                zip(predictions, training_df.source_id, training_df.dest_id, training_df.label)]
    dst_data = [(p[0], subreddit_by_id[dst], subreddit_by_id[src], label) for p, src, dst, label in
                zip(predictions, training_df.source_id, training_df.dest_id, training_df.label)]
    df = pd.DataFrame(src_data + dst_data, columns=["prediction", "src", "dst", "label"])\
        .sort_values("prediction", ascending=False)

    # Find how many true labels there were per subreddit
    true_per_subreddit = df.groupby("src").label.sum().to_frame().reset_index()
    true_per_subreddit.rename(columns={'label': 'total_true'}, inplace=True)

    # For each subreddit find the precision out of the top k values
    true_per_subreddit_at_k = df\
        .sort_values(['src', 'prediction'], ascending=False)\
        .groupby('src').head(k)\
        .groupby('src').label.sum().to_frame().reset_index()\
        .rename(columns={'label': 'sum_top_k_labels'})

    precision_at_k_per_sub = pd.merge(true_per_subreddit, true_per_subreddit_at_k, on="src")
    # For each subreddit, divide the number of true labels in the top k by the minimum of k, and the number of true
    # labels in for that subreddit in the data.
    precision_at_k_per_sub["subreddit_divisor"] = np.minimum(k, precision_at_k_per_sub.total_true)
    precision_at_k_per_sub["attainable_at_k"] = precision_at_k_per_sub.sum_top_k_labels / precision_at_k_per_sub.subreddit_divisor
    return precision_at_k_per_sub["attainable_at_k"].mean()


def symmetric_precision_at_k(model, training_df, subreddit_by_id, k):
    # For each source subreddit, find the k most likely subreddits predicted to be true. Of those, compute
    # the fraction which are true positives. This gives a precision per subreddit at k. The mean of all
    # subreddit precision at k is returned at the end.
    #
    # Treats source and destination both as sources of true positives.
    predictions = model.predict([np.array(training_df.source_id), np.array(training_df.dest_id)])

    src_data = [(p[0], subreddit_by_id[src], subreddit_by_id[dst], label) for p, src, dst, label in
                zip(predictions, training_df.source_id, training_df.dest_id, training_df.label)]
    dst_data = [(p[0], subreddit_by_id[dst], subreddit_by_id[src], label) for p, src, dst, label in
                zip(predictions, training_df.source_id, training_df.dest_id, training_df.label)]
    df = pd.DataFrame(src_data + dst_data, columns=["prediction", "src", "dst", "label"])\
        .sort_values("prediction", ascending=False)

    # For each subreddit find the precision out of the top k values
    per_subreddit_precision_at_k = df\
        .sort_values(['src', 'prediction'], ascending=False)\
        .groupby('src').head(k)\
        .groupby('src').label.sum() / k
    return per_subreddit_precision_at_k.mean()


def precision_at_k(model, training_df, subreddit_by_id, k):
    # For each source subreddit, find the k most likely subreddits predicted to be true. Of those, compute
    # the fraction which are true positives. This gives a precision per subreddit at k. The mean of all
    # subreddit precision at k is returned at the end.
    predictions = model.predict([np.array(training_df.source_id), np.array(training_df.dest_id)])

    data = [(p[0], subreddit_by_id[src], subreddit_by_id[dst], label) for p, src, dst, label in
            zip(predictions, training_df.source_id, training_df.dest_id, training_df.label)]
    df = pd.DataFrame(data, columns=["prediction", "src", "dst", "label"]).sort_values("prediction", ascending=False)

    # For each subreddit find the precision out of the top k values
    per_subreddit_precision_at_k = df\
        .sort_values(['src', 'prediction'], ascending=False)\
        .groupby('src').head(k)\
        .groupby('src').label.sum() / k
    return per_subreddit_precision_at_k.mean()


def subreddit_precision_at_k(model, training_df, subreddit_by_id, k, subreddit):
    # Compute the precision at k for just one subreddit
    # Can be used for debugging specific subreddits
    df = training_df[training_df.source_id == subreddit_by_id[subreddit]]
    return precision_at_k(model, df, subreddit_by_id, k)


def build_model(subreddit_by_id, embedding_dim):
    # Takes dictionary of subreddit_by_id, returns an untrained model
    s1_input = Input(shape=(1,), dtype='int32', name='s1_input')
    s2_input = Input(shape=(1,), dtype='int32', name='s2_input')

    # This is the main embedding matrix to be learned
    primary_embedding = Embedding(output_dim=embedding_dim, input_dim=max(list(subreddit_by_id.keys())),
                                  input_length=1, name='primary_embedding')

    s1_vec = primary_embedding(s1_input)
    s2_vec = primary_embedding(s2_input)
    # may want to use normalize true someday
    dot_product = dot([s1_vec, s2_vec], axes=2, normalize=False, name='dot_product')
    reshaped_dot = Reshape((1,))(dot_product)
    main_output = Activation("sigmoid")(reshaped_dot)

    model = Model(inputs=[s1_input, s2_input], outputs=main_output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', auc])

    print(model.summary())
    return model, primary_embedding


def train_model(model, training_df, epochs, subreddit_by_id, batch_size=32, k_values=(1, 2, 3, 5, 10)):
    # Takes a model which may or may not have already been trained, and a data frame with columns source_id, dest_id,
    # and label.
    for i in range(epochs):
        model.fit([training_df.source_id, training_df.dest_id], training_df.label, epochs=1, batch_size=batch_size)
        # Use the symmetric versions of the precision at k metric, no longer bother with the asymmetric version.
        for k in k_values:
            print("  k = {}: {}".format(k, symmetric_precision_at_k(model, training_df, subreddit_by_id, k)))
        for k in k_values:
            attainable = attainable_symmetric_precision_at_k(model, training_df, subreddit_by_id, k)
            print("  attainable k = {}: {}".format(k, attainable))
    return model


def sanity_check(embedding_matrix, subreddit_name, subreddit_to_id, id_to_subreddit, top_n):
    # convert subreddit name to id
    subreddit_id = subreddit_to_id[subreddit_name]
    # id is the column of the embedding matrix, representing the subreddit
    subreddit_embedding = embedding_matrix[subreddit_id]
    # have to find n closest subreddits using cosine similarity
    dot_prod = embedding_matrix.dot(subreddit_embedding)
    vector_norm = np.linalg.norm(embedding_matrix, axis=1)
    vector_norm = vector_norm * np.linalg.norm(subreddit_embedding)
    cos_sim = dot_prod / vector_norm
    # concatenating index column
    cos_sim = np.concatenate((np.array(range(cos_sim.shape[0])).reshape(-1, 1), cos_sim.reshape(-1, 1)), axis=1)
    # sort by the cosine similarity column
    top_n = cos_sim[np.argsort(cos_sim[:, 1])[::-1]][:top_n, :]
    return [(id_to_subreddit[top_n[i, 0]], top_n[i, 1]) for i in range(top_n.shape[0])]


def train_from_reddit_data(graph_filename):
    # todo will need to read in mapping of subreddit name to id in the near future
    if graph_filename.endswith(".gz"):
        print("Reading gzip file")
        with gzip.open(graph_filename, "rb") as f:
            graph_dict = json.load(f)
    else:
        print("Reading uncompressed file")
        with open(graph_filename, "r") as f:
            graph_dict = json.load(f)

    print("Read graph data for {}k subreddits".format(len(graph_dict.keys())//1000))
    for k in list(graph_dict.keys())[0:20]:
        print(k)

    # Filter out u/some_user
    preprocess_graph.remove_personal_subreddits(graph_dict)
    df, id_by_subreddit, subreddit_by_id = preprocess_graph.build_training_df(graph_dict)

    # Pruning and filtering, both in the absolute edge count and also on the conditional probability that two
    # subreddits are connected.
    df = preprocess_graph.prune_abs_edge(df, 20)
    df = preprocess_graph.cond_prob_edge_freq(df)
    df = preprocess_graph.prune_on_cond(df, 0.01)
    on_edge, on_cond = preprocess_graph.prune_top_n(df, 20)
    # add the label column here
    on_cond['label'] = 1

    # change the input of the negative sampler to df
    neg_sampler = NegativeSampler(on_cond)
    neg_sample_multiple = 2.0
    num_neg_samples = int(len(on_cond.index) * neg_sample_multiple)
    start_time = time()
    df_neg = pd.DataFrame(neg_sampler.sample(num_neg_samples, True), columns=["source_id", "dest_id"]).astype(int)
    df_neg['label'] = 0
    print("Requested {} negative samples, got {}".format(num_neg_samples, len(df_neg)))
    print("Took {} seconds".format(time()-start_time))

    training_df = pd.concat([df, df_neg])
    training_df["label"] = np.where(training_df.edge_count > 0, 1.0, 0.0)
    training_df = training_df.sample(frac=1).reset_index(drop=True)

    print("Total number of training rows: {}k".format(len(training_df)//1000))
    # Learn 16 dim embeddings
    model, primary_embedding = build_model(subreddit_by_id, 16)

    # Train 5 epochs
    train_model(model, training_df, 5, subreddit_by_id, batch_size=32, k_values=(1, 2, 3, 5, 10))


    # model.save_weights("model_weights.h5")
    # with open("model_architecture.json", "w") as f:
    #   f.write(model.to_json())
    #
    # from keras.models import model_from_json
    # with open("model_architecture.json", "r") as f:
    #   model = model_from_json(f.read())
    # model.load_weights("model_weights.h5")

if __name__ == '__main__':
    train_from_reddit_data("big_subreddit_graph.json.gz")
