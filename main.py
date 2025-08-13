import streamlit as st
# Import the functions from your logic file
from app import get_pdf_text, get_text_chunks, get_vectorstore, get_qa_chain

# --- STREAMLIT APP ---

def main():
    st.set_page_config(page_title="Q&A with AWS", layout="wide")
    st.title("📄 Q&A Application powered by AWS")
    st.subheader("Upload Documents, Process, and Ask Questions")

    # Initialize session state for the QA chain
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    with st.sidebar:
        st.title("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF documents and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # 1. Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        
                        # 2. Get text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # 3. Create vector store
                        vector_store = get_vectorstore(text_chunks)
                        
                        # 4. Create QA chain and store in session
                        st.session_state.qa_chain = get_qa_chain(vector_store)
                        
                        st.success("Processing Complete!")
                    except Exception as e:
                        st.error(f"ERROR: Could not process documents. Ensure AWS credentials are correct. Details: {e}")

    # Main Q&A interface
    st.header("Ask a question about your documents")
    user_question = st.text_input("Type your question here...")

    if user_question:
        if st.session_state.qa_chain:
            try:
                with st.spinner("Finding your answer..."):
                    # Invoke the chain to get a direct answer
                    response = st.session_state.qa_chain.invoke(user_question)
                    st.subheader("Answer:")
                    st.write(response)
            except Exception as e:
                st.error(f"ERROR: Could not get an answer. Details: {e}")
        else:
            st.warning("Please process your documents first.")

if __name__ == "__main__":
    main()
