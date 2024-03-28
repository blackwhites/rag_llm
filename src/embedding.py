'''
This python script contains logic to create embeddings
'''

from sentence_transformers import SentenceTransformer
from typing import List

#sentence_transformers are used to transform a list of sentences into a list of vectors
def sentence_embeddings(chunk_documents:List):

    # Load the model
    # model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Generate embeddings for a text
    embeddings = model.encode(chunk_documents)

    return embeddings #sending embeddings

#sentence_transformers are used to transform a list of sentences into a list of vectors
def sentence_embeddings_local(chunk_documents:List):

    # Load the model
    # model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")
    model = SentenceTransformer('src/models/sentence-transformers/all-MiniLM-L6-v2')

    # Generate embeddings for a text
    embeddings = model.encode(chunk_documents)

    return embeddings #sending embeddings


