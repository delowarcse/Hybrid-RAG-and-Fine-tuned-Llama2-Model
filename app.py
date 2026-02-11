# app.py - Main application code for the hybrid RAG + fine-tuned Llama2 system

from langchain_community.document_loaders import PyPDFLoader  # Use PDF loader as per assignment
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load and process the full Llama2 research paper PDF
# Assume the PDF file is named "llama2paper.pdf" in the same directory
pdf_loader = PyPDFLoader("llama2paper.pdf")
documents = pdf_loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Added overlap for better context
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")  # Persist for efficiency

# Load fine-tuned Llama2 model via Ollama (assuming "AIresearcher:latest" is the fine-tuned version)
llm = Ollama(model="AIresearcher:latest")

# Custom prompt template for hybrid RAG + fine-tuned model interaction
# The fine-tuned model provides specialized knowledge, while RAG supplies document context
retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an AI researcher assistant specialized in large language models like Llama2.
    Use your fine-tuned knowledge on LLMs combined with the provided context from the Llama2 paper to answer questions accurately.
    If the context doesn't cover the query, rely on your trained knowledge but note any uncertainties.
    Context: {context}
    """),
    ("human", "{input}"),
])

# Create document chain and retrieval chain
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retriever = db.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Function to query the system (used in Streamlit)
def query_hybrid_system(input_query):
    response = rag_chain.invoke({"input": input_query})
    return response["answer"]