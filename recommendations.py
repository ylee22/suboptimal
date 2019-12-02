import numpy as np
from keras.models import model_from_json
import json
import gzip


def recommendations(subreddit_name, n, embedding_matrix, id_by_subreddit, subreddit_by_id):
    # convert subreddit name to id
    subreddit_id = id_by_subreddit[subreddit_name]
    # id is the column of the embedding matrix, representing the subreddit
    subreddit_embedding = embedding_matrix[subreddit_id]
    # have to find n closest subreddits using cosine similarity
    dotprod = embedding_matrix.dot(subreddit_embedding)
    vectornorm = np.linalg.norm(embedding_matrix, axis=1)
    vectornorm = vectornorm*np.linalg.norm(subreddit_embedding)
    cossim = dotprod#/vectornorm
    # concatenating index column
    cossim = np.concatenate((np.array(range(cossim.shape[0])).reshape(-1,1), cossim.reshape(-1,1)), axis=1)
    # sort by the cosine similarity column
    # grab the most similar (recommended subreddits)
    top_n = cossim[np.argsort(cossim[:, 1])[::-1]][:n, :]
    top_n = [(subreddit_by_id[top_n[i, 0]]) for i in range(top_n.shape[0])]
    # grab the least similar (least recommended subreddits)
    bot_n = cossim[np.argsort(cossim[:, 1])[::-1]][-n:, :]
    bot_n = [(subreddit_by_id[bot_n[i, 0]]) for i in range(bot_n.shape[0])]
    return top_n, bot_n


def get_embedding():
    # load json and create model
    json_file = open('model_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_weights.h5")
    embedding_matrix = loaded_model.get_layer('primary_embedding').get_weights()[0]
    return embedding_matrix


def get_subreddit_dicts():
    with open('subreddit_by_id.json') as json_file:
        subreddit_by_id = json.load(json_file)

    id_by_subreddit = {v: int(k) for k, v in subreddit_by_id.items()}
    subreddit_by_id = {int(k): v for k, v in subreddit_by_id.items()}
    return id_by_subreddit, subreddit_by_id


def subreddit_embeddings(subreddit_names, embedding_matrix, id_by_subreddit):
    # subreddit indices
    subred = [id_by_subreddit[subred] for subred in subreddit_names]

    # include a random sampling of the embedding to give the tsne more structure
    rest_idx = set(range(embedding_matrix.shape[0])) - set(subred)
    rand_idx = np.random.choice(list(rest_idx), size=200, replace=False)

    #subreddit indices are at the top
    idx = subred + list(rand_idx)
    return embedding_matrix[idx]


def load_subreddit_text():
    with gzip.GzipFile('subreddit_texts.json.gz', 'r') as fin:  # 4. gzip
        json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)

    json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
    text_dict = json.loads(json_str)  # 1. data
    return text_dict
