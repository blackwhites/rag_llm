
'''
This is file works upto creation of vector store 
'''
#Imports 
from data_collection_parsing import get_document
from chunking import do_chunking
from embedding import sentence_embeddings
from search_vector_creation import create_and_store_data_faiss


# Do data collection 
filenames = ['new.docx'] #list containing all document names
documents = get_document(filenames)

# Do data chunking
# documents contain list of paragraphs stored as doc format 
#so we call each doc (indirectly paragraph) to do chunking 
chunked_documents = []
for doc in documents:
    chunks = do_chunking(doc.text)
    chunked_documents.extend(chunks)

# Do embedding
embedded_documents = sentence_embeddings(chunk_documents = [doc.page_content for doc in chunked_documents])
'''
we have paragraphs in the document 
'''

#Creating Vector database
write_index_for_local = True
create_and_store_data_faiss(chunked_documents,embedded_documents,write_index_for_local)

'''
we need to integrate this with elastic search or other store if we are intrested to add text data into vector store

'''
# Do create a pipline 
'''
    First Query Search from User input 
    Take top 3 retreivers
    Send these 3 to Model and ask for final one answer
    Send this final one answer to User
'''

