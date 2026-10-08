import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

def main() -> None:
    print("Ingesting data into pinecone")
    # loding the data from the file
    loader = UnstructuredLoader(file_path="src/langchain_course/mediumblog1.txt")
    data = loader.load()
    print(f"Loaded {len(data)} documents")
    # splitting the data into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_data = splitter.split_documents(data)
    print(f"Split {len(split_data)} documents")
    # creating embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    # creating vector store
    vector_store = PineconeVectorStore.from_documents(
        split_data,
        embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )
    print("Data ingested into pinecone successfully")


if __name__ == "__main__":
    main()