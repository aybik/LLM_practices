import fitz  
import numpy as np
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
#from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text if text.strip() else None  # Avoid empty documents

def load_and_process_pdfs(pdf_files):
    """Loads multiple PDFs and extracts valid text."""
    documents = [extract_text_from_pdf(pdf) for pdf in pdf_files if extract_text_from_pdf(pdf)]
    return [doc for doc in documents if isinstance(doc, str) and doc.strip()]  # Ensure valid text

def split_text_into_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Splits text into smaller chunks for efficient processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = []
    for doc in documents:
        texts.extend(text_splitter.split_text(doc))
    return texts

def create_faiss_vectorstore(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generates embeddings and stores them in a FAISS vector database."""
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    documents = [Document(page_content=text) for text in texts]  
    vectorstore = FAISS.from_documents(documents, embedding_model)
    print("✅ FAISS vector store successfully created!")
    return vectorstore

def run_pipeline(pdf_files, query):
    llm = Ollama(model="deepseek-r1:1.5b") 
    documents = load_and_process_pdfs(pdf_files) 
    texts = split_text_into_chunks(documents)  
    vectorstore = create_faiss_vectorstore(texts)  
    
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.run(query)
    return response