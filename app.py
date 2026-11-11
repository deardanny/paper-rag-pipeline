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

def read_cli_input(prompt):
    """Return None when the user closes or cancels an input prompt."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return None

def main():
    pdf_path = read_cli_input("Enter PDF path (e.g. paper.pdf): ")
    if pdf_path is None:
        return
    pdf_path = os.path.expanduser(pdf_path)
    if not os.path.isfile(pdf_path):
        print("PDF file not found. Enter the path to an existing local PDF file.")
        return

    print("Loading paper...")
    chunks = load_paper(pdf_path)
    print(f"Split into {len(chunks)} chunks")
    if not chunks:
        print("No text was extracted from this PDF. Try a PDF with selectable text.")
        return

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
        query = read_cli_input("Question: ")
        if query is None or query.lower() == "quit":
            break
        if not query:
            print("Please enter a question.")
            continue
        answer = chain.invoke(query)
        print(f"\nAnswer: {answer.content}\n")

if __name__ == "__main__":
    main()
