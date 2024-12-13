

import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Setup directories
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

# Contextualize question system prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Define the system prompt
system_prompt = '''You are a helpful bot assistant that helps people to answer questions based on pdf files.
"\n\n"
{context}
'''

qa_system_prompt = system_prompt

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Streamlit app
st.title("PDF-based Chat Assistant")
st.write("Ask questions based on PDF files analyzed by the assistant.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user query
query = st.text_input("You:", key="user_input")
if st.button("Send") and query:
    # Process the query through the retrieval chain
    result = rag_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
    
    # Display the AI response
    st.write(f"AI: {result['answer']}")

    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(SystemMessage(content=result["answer"]))
