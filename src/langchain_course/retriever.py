
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-5.2")

vectorstore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = """
Use the following pieces of context to answer the question.
If you don't know the answer, just say that you don't know.
Keep the answer concise and based on the context.

Context: {context}
Question: {question}

Provide a detailed answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM invocation (without RAG)")
    print("=" * 70)
    
    question = "What is the purpose of machine learning?"
    print(f"Question: {question}")
    response = llm.invoke([HumanMessage(content=question)])
    print(f"Response: {response.content}")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Basic RAG pipeline (standalone retriever)")
    print("=" * 70)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(question)
    print(f"Response: {response}")
    print("=" * 70)