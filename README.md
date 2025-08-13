# AWS RAG Q&A Application

This project is a Question-Answering application that uses the Retrieval-Augmented Generation (RAG) pattern, powered by Amazon Web Services (AWS). It allows users to upload PDF documents, process them, and ask questions based on their content. The application is built with Python, Streamlit, and LangChain, and is containerized with Docker for portability.

---

## 🏛️ Architecture

The application is built around a few key cloud services and libraries:

* **Frontend:** `Streamlit` is used to create a simple, interactive web interface for file uploads and Q&A.
* **Backend Logic:** `Python` with the `LangChain` framework orchestrates the RAG pipeline.
* **Data Storage:** `Amazon S3` acts as a secure, private object store for the source PDF documents.
* **AI Models:** `Amazon Bedrock` provides access to powerful foundation models for creating text embeddings and generating answers.
* **Deployment:** `Docker` is used to containerize the application, ensuring it runs consistently in any environment.

---

## 🛠️ Key Components & Setup

### 1. Amazon S3 (Simple Storage Service)

We used S3 as our secure "filing cabinet" for the knowledge base.

* **Bucket Creation:** A private S3 bucket was created in the `ap-south-1 (Mumbai)` region to store the user-uploaded PDF documents.
* **Security:** The bucket was configured with **"Block all public access"** enabled. This is a critical security measure to ensure our source data is never exposed to the public internet. Access is granted exclusively through secure IAM credentials.

### 2. AWS IAM (Identity & Access Management)

To ensure our application connected to AWS securely, we followed the **Principle of Least Privilege**.

* **Application-Specific User:** We created a dedicated IAM user (`rag-app-user`) specifically for our application. This user was configured for **programmatic access** only (not for console login).
* **Permissions:** We attached AWS-managed policies (`AmazonS3FullAccess` and `AmazonBedrockFullAccess`) to this user. This granted the application the exact permissions it needed to function and nothing more.
* **Credentials:** We generated an **Access Key ID** and a **Secret Access Key** for this user. These keys were stored locally in a `.env` file, which is never committed to source control.

### 3. Amazon Bedrock & LangChain Runnable

This is the core AI engine of our application.

* **Embeddings:** We used the `amazon.titan-embed-text-v1` model via Bedrock to convert our text chunks into numerical vector embeddings.
* **Answer Generation:** We used the `anthropic.claude-v2` model via Bedrock to generate human-like answers based on the context retrieved from our documents.
* **LCEL Runnable:** We implemented the RAG logic using a modern **LangChain Expression Language (LCEL)** runnable sequence. This chain takes a user's question, retrieves relevant context from a `FAISS` vector store, and pipes this information into a prompt that instructs the Claude model to generate a final answer.

### 4. Docker Containerization

To make the application portable and easy to run, we used Docker.

* **Dockerfile:** We created a `Dockerfile` that defines the blueprint for our application's environment.
* **Process:** The `Dockerfile` starts from a base Python image, installs all dependencies from `requirements.txt`, copies our application code (`main.py`, `app.py`), and defines the command `streamlit run main.py` to start the server.
* **Result:** This process creates a self-contained image that can be run on any machine with Docker installed, solving all dependency and environment issues.

---

## 🚀 How to Run the Project

### Prerequisites
* Docker Desktop installed and running.
* A `.env` file in the root directory with your AWS credentials.

### Steps
1.  **Build the Docker image:**
    ```bash
    docker build -t my-rag-app .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 my-rag-app
    ```

3.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8501`.

