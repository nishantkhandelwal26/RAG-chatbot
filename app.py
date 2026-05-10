import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload a PDF and ask questions about its content.")

api_key = st.text_input("Enter your Groq API Key", type="password")


if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")

    session_id = st.text_input("Session Id", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}
    
    upload_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if upload_files:
        documents = []
        for uploaded_file in upload_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed otherwise return it as it is."
        )


        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
            ]
        )


        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Prompt template for question answering
        system_prompt = (
            "You are an intelligent AI assistant specialized in answering questions. "
            "Use the retrieved context provided below to generate an accurate response. "
            "If the answer cannot be determined from the context, clearly state that you do not know. "
            "Keep the response brief, relevant, and limited to a maximum of three sentences.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
            ]
        )

        question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer")

        user_input = st.text_input("Ask a question about the uploaded PDFs:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {
                    "input": user_input
                },
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )

            st.write(st.session_state.store)
            st.success(response['answer'])
            st.write("Chat history for this session:", session_history.messages)
        
else:
    st.warning("Please enter your Groq API Key to use the application.")