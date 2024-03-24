# '''
# This python script contains logic for getting the hugging face model for 
# '''
#imports
from embedding import sentence_embeddings
from search_vector_creation import search_faiss
from langchain import PromptTemplate, LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



#Ask user query
user_input = input("Please enter the query")
question = user_input #for ease if in case i double use both

'''
FUTURE SCOPE :  WE CAN DO QUERY EXPANSION (GENERATION OF QUERIES USING LLM MODEL IT SELF :) )
'''

#Get the search retrivals from database like top 3
'''
converting query to embedded vector as faiss search only on vectors 
(though vector represents multi modal data, it dont care it search/saves only vectors)

'''
index_name = 'technical_engineering_vector_store.faiss'
question_vector = sentence_embeddings([question])[0]
retreived_results = search_faiss(index_name,question_vector)
print(retreived_results) #print


#Add template
prompt_template ='''
Your an assitant of technical engineer. 
You will take the context from the user as context
also you will have option to look on what question  you are trying to answer based on the context provided.

Restrictions :  Your restricted to use your own knowledge or your restricted to hallucinate , Try to answer from the context for question

answer should include similarity score between context and answer generated from chatgpt by having 
(answer similarity_score : score) at end of answer
context:{context}
Question : {question}
'''

#Config Model
prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])


#Convey prompt to model and input the obtained 3 retreivals
from transformers import AutoTokenizer, MistralForCausalLM

model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def load_model(model):
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

rag_chain = (
    {"context": 'Result : '.join(retreived_results), "question": RunnablePassthrough()} 
    |prompt
    |load_model(model)
    | StrOutputParser()
)

rag_chain.invoke("What is poor managment issues?")
