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

load_dotenv()

# --- AWS CONFIG ---
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- AWS CLIENTS ---
s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# ---------- S3 FUNCTIONS ----------

def upload_pdf_to_s3(pdf_file):
    """Uploads PDF to S3 and returns S3 key"""
    s3_key = f"uploads/{pdf_file.name}"
    s3_client.upload_fileobj(pdf_file, S3_BUCKET_NAME, s3_key)
    return s3_key


def get_pdf_text_from_s3(s3_keys):
    """Reads PDFs from S3 and extracts text"""
    text = ""

    for key in s3_keys:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        pdf_reader = PdfReader(obj["Body"])

        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


# ---------- RAG FUNCTIONS ----------

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_client
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_qa_chain(vectorstore):
    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock_client
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 8}
    )

    prompt_template = """Given the following context and a question, generate a concise and relevant answer based primarily
      on the "response" section in the source document context. If the question is a greeting (like "hi", "hello", "hey"), respond politely but briefly. If the question is outside the scope of the context or no relevant information is found,
      politely say that you don't have the information but you are here to help. Keep your answer short and to the point. 
      CONTEXT: {context} QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
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





