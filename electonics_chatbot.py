from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys
import time

from langchain.document_loaders import DirectoryLoader

# local_directory = "./local_llama2_model"
# tokenizer = AutoTokenizer.from_pretrained(local_directory)
# model = AutoModelForCausalLM.from_pretrained(local_directory)

# Load the local Llama 2 model
model_path = r"C:\Users\ShibashishNayak\Videos\Models\electronics-chatbot\local_llama2_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# llm = CTransformers(
#         # model = "BioMistral-7B.Q5_K_M.gguf",
#         # model = "BioMistral-7B.Q8_0.gguf",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens = 600,
#         temperature = 0.01,
#         context_length= 700,
#         )

# Create a pipeline for text generation
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=600,
    temperature=0.01,
)

# Wrap the pipeline with HuggingFacePipeline for LangChain
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db_path = "faiss_vector_db"
def create_vector_db():
    # loader = CSVLoader(file_path="new_dataset_9apr.csv", source_column='Question', encoding="utf-8")
    # loader = CSVLoader(file_path="merged_data_set_medicalDevice.csv", source_column='Question', encoding="utf-8")
    loader = UnstructuredExcelLoader(file_path="Electric Vehicle.xlsx", source_column='Tag', encoding="utf-8")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    docsearch.save_local(vector_db_path)
    return ({'message': 'Vector DB creation completed'})
# call the function if data set changed or new data added
print("vector Db started")
# create_vector_db() 
vectordb = FAISS.load_local(vector_db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
print("vector Db loaded")
retriever = vectordb.as_retriever(score_threshold=0.7)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
print("Enter your Query...  type exit or Exit to stop the chat")
while True:
    
    chat_history = []
    print(" ")
    query = input(f"You: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue

    start_time = time.time()

    result = qa({"question":query, "chat_history":chat_history})
    response = result['answer']

    end_time = time.time()  # Capture end time
    response_time = end_time - start_time  # Calculate response time

    print("Chatbot: ", result['answer'])
    print(f"Response Time: {response_time:.2f} seconds")