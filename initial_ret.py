from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


def create_embeddings(collection, queries):
    # Initialize the model
    model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("device: ", device)
    print("Type of dataset: ", type(collection))

    # Encode the documents
    doc_embeddings = model.encode(collection['text'].tolist(), show_progress_bar=True)

    # Encode the queries
    query_embeddings = model.encode(queries['query'].tolist(), show_progress_bar=True)

    with open('doc_embeddings.pkl', 'wb') as f:
        pickle.dump(doc_embeddings, f)

    with open('query_embeddings.pkl', 'wb') as f:
        pickle.dump(query_embeddings, f)

    print(doc_embeddings.shape)
    print(query_embeddings.shape)

    return doc_embeddings, query_embeddings


def perform_initial_ret(doc_embeddings, query_embeddings, queries, collection):

    top_n = 5
    initial_retrieval_results = {}

    for idx, query_emb in enumerate(query_embeddings):
        # Compute similarities with all documents
        similarities = cosine_similarity([query_emb], doc_embeddings)[0]

        # Get the indices of the top N documents
        top_doc_indices = np.argsort(similarities)[::-1][:top_n]

        # Store the results along with doc_id and similarity score
        qid = queries.iloc[idx]['qid']
        initial_retrieval_results[qid] = [(collection.iloc[doc_idx]['docid'], similarities[doc_idx]) for doc_idx in
                                          top_doc_indices]

    return initial_retrieval_results


def perform_initial_ret_old(doc_embeddings, query_embeddings, queries):

    top_n = 5
    initial_retrieval_results = {}

    for idx, query_emb in enumerate(query_embeddings):
        # Compute similarities with all documents
        similarities = cosine_similarity([query_emb], doc_embeddings)[0]

        # Get the indices of the top N documents
        top_doc_indices = np.argsort(similarities)[::-1][:top_n]

        # Store the results
        initial_retrieval_results[queries.iloc[idx]['qid']] = top_doc_indices

    return initial_retrieval_results