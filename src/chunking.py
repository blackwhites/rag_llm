'''
This Python Script contains logic to chunk the documents / Text Parsed information using Langchains and other methods
'''

#import langchain with overlaps / size
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

def do_chunking(paragraph):
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )   
    texts = text_splitter.create_documents([paragraph])
    return texts



