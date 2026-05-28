from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdf():
    loader=PyPDFLoader('data/wellarchitected-framework.pdf')
    docs=loader.load()
    return docs


def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,
    )
    return text_splitter.split_documents(docs)


if __name__ == "__main__":
    docs = load_pdf()
    chunks = split_text(docs)
    print(f'Number of chunks: {len(chunks)}')
    # for chunk in chunks:
    #     print(chunk.page_content)
    #     print("Page:", chunk.metadata)
    #     print("---")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("embeddings/faiss_index")
    print("Index created and saved successfully.")
    
