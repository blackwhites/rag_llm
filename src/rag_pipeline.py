'''
This python script contains logic for getting the hugging face model for 

FUTURE SCOPE :  WE CAN DO QUERY EXPANSION (GENERATION OF QUERIES USING LLM MODEL IT SELF :) )
'''


from embedding import sentence_embeddings,sentence_embeddings_local
from search_vector_creation import search_faiss
from langchain import PromptTemplate, LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, MistralForCausalLM,AutoModelForCausalLM
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

class TechnicalEngineeringAssistant:
    def __init__(self, index_name,model_name, is_local_model,custom_prompt=None):
        self.index_name = index_name
        self.is_local_model = is_local_model
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        if custom_prompt:
            self.prompt_template = custom_prompt
        else:
            self.prompt_template =  '''[INST]
        You an assitant of technical engineer.
        You will be given context and question.
        you need to give answer for question.
        Your answer should be as precise as possible and should only come from the context.
       
        Restrictions :  Your restricted to use your own knowledge or your restricted to hallucinate , Try to answer from the context for question

        answer should include similarity score between context and answer generated from you by having
        (answer similarity_score : score) at end of answer ans stop after this score.
        context:{context}
        Question : {question}
        [/INST]'''

    def get_retrieved_results(self, question):
        #Get the search retrivals from database like top 3
        '''
        converting query to embedded vector as faiss search only on vectors 
        (though vector represents multi modal data, it dont care it search/saves only vectors)

        '''
        if self.is_local_model:
            question_vector = sentence_embeddings_local([question])[0]
        else:
            question_vector = sentence_embeddings([question])[0]
        return search_faiss(self.index_name, question_vector)

    def load_model_(self, prompt):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids, max_length=30)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_response(self, question,custom_prompt=None):
        
        retreived_results = self.get_retrieved_results(question)
        prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        rag_chain = (
            {"context": 'Result : '.join(retreived_results), "question": RunnablePassthrough()}
            | prompt
            | self.load_model
            | StrOutputParser()
        )
        return rag_chain.invoke(question)
    def get_response_from_local_model(self,question,custom_prompt=None):
        if custom_prompt:
            self.prompt_template = custom_prompt
        retreived_results = self.get_retrieved_results(question)
        print("retreived result:\n")
        print(retreived_results)
        print('\n')
        
        prompt = self.prompt_template.format(context= 'Result : '.join(retreived_results), question=question)

        # Point to the local server
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        # Create a system message with the formatted prompt
        system_message = {"role": "system", "content": prompt}
        completion = client.chat.completions.create(
            model = self.model_name,
        messages=[
            system_message
        ],
        temperature=0.7,
        )

        return completion.choices[0].message.content



# model_name = "mistralai/Mistral-7B-v0.1"
model_name = 'mistralai_mistral-7b-instruct-v0.2'
index_name = 'technical_engineering_vector_store.faiss'
run_local_model = True
custom_prompt = '''
system:
You are an AI assistant that helps users answer questions given a specific context. You will be given a context and asked a question based on that context. Your answer should be as precise as possible and should only come from the context.
Please add citation after each sentence when possible in a form "(Source: citation)".

user:
{{contexts}}
Human: {{question}}
AI:'''

# Create an instance of the class and get a response
def rag_pipeline(user_input,model_name,index_name,run_local_model):
    assistant = TechnicalEngineeringAssistant(index_name=index_name,model_name=model_name,is_local_model=run_local_model,custom_prompt=None)
    if run_local_model :
        response = assistant.get_response_from_local_model(user_input)
    else:
        response = assistant.get_response(user_input)
    print(response )
    print('\n')
    return response

while True:
    # Ask user query
    user_input = input("\nPlease enter the query: ")
    rag_pipeline(user_input,model_name,index_name,run_local_model)
    print('###################################################')
