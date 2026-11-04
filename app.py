import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

def load_paper(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def main():
    pdf_path = input("Enter PDF path (e.g. paper.pdf): ")

    print("Loading paper...")
    chunks = load_paper(pdf_path)
    print(f"Split into {len(chunks)} chunks")

    print("Building vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    print("\nReady! Ask anything about the paper. (type 'quit' to exit)\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            print("Please enter a question.")
            continue
        answer = chain.invoke(query)
        print(f"\nAnswer: {answer.content}\n")

if __name__ == "__main__":
    main()