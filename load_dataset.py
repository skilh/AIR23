import pandas as pd
import utils

file_path = './data/'
subset_size = 10000


def load_collection():
    try:
        collection = pd.read_csv(file_path + 'collection.tsv', sep='\t', header=None, names=['docid', 'text'])
        return collection
    except FileNotFoundError:
        print("Collection file not found!")
        return None


def load_queries_train():
    try:
        queries = pd.read_csv(file_path + 'queries.train.tsv', sep='\t', header=None, names=['qid', 'query'])
        return queries
    except FileNotFoundError:
        print("Queries file not found!")
        return None


def load_qrels_train():
    try:
        qrels = pd.read_csv(file_path + 'qrels.train.tsv', sep='\t', header=None, names=['qid', 'q0', 'docid', 'rel'])
        return qrels
    except FileNotFoundError:
        print("Qrels file not found!")
        return None


def load_subset():
    try:
        collection = load_collection()
        queries = load_queries_train()
        qrels = load_qrels_train()

        if collection.empty or queries.empty:
            raise ValueError("One or more datasets are empty.")

        if qrels.empty:
            raise ValueError("Qrels empty")

        # Preprocess
        collection['text'] = collection['text'].apply(utils.clean_text)
        queries['query'] = queries['query'].apply(utils.clean_text)

        # Filter the collection to include only docs present in qrels
        filtered_collection = collection[collection['docid'].isin(qrels['docid'])]
        # Filter the queries to include only queries present in qrels
        filtered_queries = queries[queries['qid'].isin(qrels['qid'])]
        reduced_collection = filtered_collection.sample(n=subset_size, random_state=1)
        updated_qrels = qrels[qrels['docid'].isin(reduced_collection['docid'])]
        reduced_queries = filtered_queries.sample(n=subset_size, random_state=1)
        updated_qrels = updated_qrels[updated_qrels['qid'].isin(reduced_queries['qid'])]

        return reduced_collection, reduced_queries, updated_qrels

    except Exception as e:
        (print(f"An error occurred: {e}"))
        return None, None, None


if __name__ == '__main__':

    collection, queries, qrels = load_subset()

    if collection is not None:
        print(collection.head(5))
    if queries is not None:
        print(queries.head(5))
    if qrels is not None:
        print(qrels.head(5))

    print(len(collection))
    print(len(queries))
    print(len(qrels))

    print(type(collection))
    print(type(queries))
    print(type(qrels))

