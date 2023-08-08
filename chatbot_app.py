import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def main():
    load_dotenv()
    st.set_page_config(page_title="PDFs chatbot", page_icon='📕')

    #Sidebar
    #Title and documents uploader
    with st.sidebar:
        st.title("🤖💬📕 PDFs chatbot")
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents 📁", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing..."):
                #Get text from pdfs
                raw_text = get_pdf_text(pdf_docs)

                #Get the text chunks 
                text_chunks = get_chunks(raw_text)

                #Create vector store
                vectorestore = get_vector_store(text_chunks)
    
    #Create some space between the uploader and contact info
    with st.sidebar.container():
        st.text("")
        st.markdown("***")

    #Contact info
    with st.sidebar.container():
        st.subheader(
            "A Streamlit web app by [Hiram Cortes](https://www.linkedin.com/in/hdcortesd/)"
            )
        st.markdown("You can find the source code on my [GitHub](https://github.com/MaxPower14)")


if __name__ == '__main__':
    main()