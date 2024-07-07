import os
import shutil
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
# os.environ["LANGCHAIN_API_KEY"] =


def get_docs(paths):
    loaders = []
    for path in paths:
        loaders.append(PyPDFLoader(path))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    splits = text_splitter.split_documents(docs)
    return splits


def getQA(paths):
    template = """
    Your job is trying to answer the questions of user base on given context,
    The answer should not longer than 5 sentence. You should keep the response as concise as possiple.
    If you found that the context is not relevant to the question, tell them you don't found any relevant information 
    about their question on given documents. keep the answer natural.
    
    {context}
    Question: {question}
    Helpful Answer:"""
    PROMPT = PromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="full_documents", embedding_function=GPT4AllEmbeddings()
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    retriever.add_documents(get_docs(paths))

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    with st.sidebar:
        st.text("You are free to ask now!")
    return qa


# def set_vector_store(docs, embed_model, save_dir):
#     faiss_db = FAISS.from_documents(docs, embed_model)
#     faiss_db.save_local(save_dir)


st.title("RAG Demo")
paths = []
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="chatbot_api_key", type="password"
    )

    st.subheader("Your documents")
    uploaded_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if uploaded_files is not None:
        destination_path = ".\docs"
        for uploaded_file in uploaded_files:
            destination_file_path = os.path.join(destination_path, uploaded_file.name)
            paths.append(destination_file_path)
            with open(destination_file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

        # if st.button("Process"):
        #     with st.spinner("Processing"):
    #                 raw_text = get_docs(paths)
    #                 docs = split_documents(raw_text)
    #                 set_vector_store(docs = docs, embed_model=GPT4AllEmbeddings(), save_dir='faiss_index')

    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I assist you today?"}
        ]

    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

try:
    qa = getQA(paths=paths)
except Exception:
    st.warning("⚠ Please upload your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if input := st.chat_input("How may I assist you today?"):
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    st.secrets.openai_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": input})

    with st.chat_message("user"):
        st.markdown(input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in qa.run(st.session_state.messages[-1]["content"]):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
