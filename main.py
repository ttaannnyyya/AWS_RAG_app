import streamlit as st
from app import (
    upload_pdf_to_s3,
    get_pdf_text_from_s3,
    get_text_chunks,
    get_vectorstore,
    get_qa_chain
)

def main():
    st.set_page_config(page_title="Q&A with AWS", layout="wide")
    st.title("📄 Q&A Application powered by AWS")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    with st.sidebar:
        st.title("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click Process",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Uploading to S3 and processing..."):
                    try:
                        s3_keys = []
                        for pdf in pdf_docs:
                            key = upload_pdf_to_s3(pdf)
                            s3_keys.append(key)

                        raw_text = get_pdf_text_from_s3(s3_keys)
                        chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(chunks)

                        st.session_state.qa_chain = get_qa_chain(vectorstore)
                        st.success("Documents processed successfully!")

                    except Exception as e:
                        st.error(f"Error: {e}")

    st.header("Ask a question")
    user_question = st.text_input("Type your question here")

    if user_question:
        if st.session_state.qa_chain:
            with st.spinner("Generating answer..."):
                answer = st.session_state.qa_chain.invoke(user_question)
                st.write(answer)
        else:
            st.warning("Please process documents first")

if __name__ == "__main__":
    main()



