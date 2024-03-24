'''
This python script will 
i. create a new Vector search 
ii. upsert the data into the vector database.
iii. query the vector database

'''

#imports
from elasticsearch import Elasticsearch
import faiss
import numpy as np
import json


#faiss only

def create_and_store_data_faiss(texts,vectors,locally_save):
    # Initialize Faiss index
    dimension = 384  # the dimension of your vectors
    index = faiss.IndexHNSWFlat(dimension, 32)  # using HNSW index with 32 neighbors per level

    # Set index parameters
    index.hnsw.efConstruction = 200  # number of edges per node during construction
    index.hnsw.M = 16  # maximum number of neighbors to consider during search

    # Add vectors to index
    for idx, vector in enumerate(vectors):
        index.add(np.array([vector]))

    index.nprobe = 10  # number of cells probed in quantizer search

    if locally_save :
        faiss.write_index(index,'src/database/technical_engineering_vector_store.faiss')
        
        #we also need to save chunk data because we already know that we dont have opiton to save text
        text_data = {}
        for id,text in enumerate([doc.page_content for doc in texts]):
            text_data[id] = text

        # save the dictionary to disk as a JSON file
        with open("src/database/technical_engineering_vector_store.json", "w") as f:
            json.dump(text_data, f)


#searching faiss
def search_faiss(index_name,question_vector):

    #reading faiss file
    index = faiss.read_index(f'src/database/{index_name}')

    #reading text file
    #load the dictionary from the JSON file
    with open(f"src/database/{index_name.split('.')[0]}.json", "r") as f:
        chunks_data = json.load(f)
        
    query_vector = question_vector

    # perform the search
    k = 3  # number of top results to retrieve
    distances, vector_ids = index.search(np.array([query_vector]), k)

    # retrieve the corresponding text and metadata for the top results
    # (assuming you have stored the text and metadata in a separate data structure)
    top_results = []
    for id in vector_ids[0]:
        # retrieve the text  for the result using the vector ID
        text = chunks_data[f'{id}']
        # top_results.append((text, distances[0][id]))
        top_results.append(text)
    return top_results


# Function to create and store data
def create_and_store_data(texts, vectors): #you can add metadatas
    # Initialize Elasticsearch client
    es = Elasticsearch(hosts="localhost:9200")

    # Initialize Faiss index
    dimension = 384  # the dimension of your vectors
    index = faiss.IndexFlatL2(dimension)  # using L2 distance for simplicity

    # Store data in Elasticsearch
    for idx, (text, metadata) in enumerate(texts):
        doc = {"text": text, "vector_id": idx}
        es.index(index="rag_technical_engineering_index", body=doc)

    # Store vectors in Faiss
    for idx, vector in enumerate(vectors):
        index.add(np.array([vector]))

    # After all vectors are added, the index is trained with them
    faiss.normalize_L2(index.ntotal)
    index.nprobe = 10  # number of cells probed in quantizer search