import os
import boto3
from dotenv import load_dotenv
from pypdf import PdfReader

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Bedrock
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# --- AWS Configuration ---
# These credentials will be loaded from your .env file
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# --- CORE FUNCTIONS ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Creates a FAISS vector store from text chunks and Bedrock embeddings."""
    bedrock_client = boto3.client(
        service_name="bedrock-runtime", region_name=aws_region,
        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
    )
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1", client=bedrock_client
    )
    return FAISS.from_texts(texts=text_chunks, embedding=bedrock_embeddings)

def get_qa_chain(vectorstore):
    """Creates a Question-Answering chain using the provided vector store."""
    llm = Bedrock(
        client=boto3.client(
            service_name="bedrock-runtime", region_name=aws_region,
            aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
        ),
        model_id="anthropic.claude-v2"
    )
    
    retriever = vectorstore.as_retriever(score_threshold=0.7)
    
    prompt_template = """Given the following context and a question, generate a concise and relevant answer based primarily based on the source document context.
If the question is a greeting (like "hi", "hello", "hey"), respond politely but briefly.
If the question is outside the scope of the context or no relevant information is found, politely say that you don't have the information but you are here to help.
Keep your answer short and to the point.
CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    return rag_chain




