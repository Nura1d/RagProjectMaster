import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document  # Import Document class
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Directories
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(current_dir, "db_pdf")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"PDF directory: {pdf_dir}")
print(f"Persistent directory: {persistent_directory}")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(
            f"PDF directory '{pdf_dir}' does not exist. Please provide a valid PDF directory."
        )

    # Load PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    documents = []
    for book_file in pdf_files:
        file_path = os.path.join(pdf_dir, book_file)
        print(f"Processing file: {file_path}")

        # Extract text from the PDF file
        reader = PdfReader(file_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()

        # Add the extracted text as a Document with metadata
        documents.append(Document(page_content=pdf_text, metadata={"source": book_file}))

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"  # Update to a valid embedding model if needed
    )
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")