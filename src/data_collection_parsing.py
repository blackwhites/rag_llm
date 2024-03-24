#This python script contains functions that collect data in form of .txt,.doc and parse them into paragraphs or full text fr

#Imports
from annotated_types import doc
from docx import Document
import os

#obtaining doucment and returning list of paragraphs
def obtain_paragraphs(doc):  #we can add extra logic here for obtaining paragraphs and metadata for each paragraph and convert to Doc
    return doc.paragraphs

#funciton for obtaining document
def get_document(filenames: list):
    file_paths = [] #for storing file paths
    file_names = [] #for storing file names for reducing duplicacy
    overall_paragraphs = []

    for filename in filenames:
        file_name = filename
        file_path = os.path.join('data/doc/', file_name)

        # checking duplicates in files and skipping it 
        if file_name in file_names:
            continue

        #obtaining paragraphs from document files
        doc = Document(file_path)   #reading document file 
        paragraphs = obtain_paragraphs(doc) #getting paragraphs
        overall_paragraphs.extend(paragraphs) #adding all paragraphs

        #appending filepaths and file names for all documents to remove dependecy
        file_paths.append(file_path)
        file_names.append(file_name)

    return overall_paragraphs

    

        





        



