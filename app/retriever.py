from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np

def load_index():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("embeddings/faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore, embeddings


def retrieve(query, vectorstore, k=3):
    return vectorstore.similarity_search(query, k=k)


def retrieve_raw(query, embedding_model, vectorstore, k=3):
    query_embed = embedding_model.embed_query(query)
    numpy_arr = np.array(query_embed).reshape(1, -1)
    distances, indices = vectorstore.index.search(numpy_arr, k=k)
    return distances, indices


if __name__ == "__main__":
    vectorstore, embedding_model = load_index()
    query = "What are the five pillars of the AWS Well-Architected Framework?"
    query_embed = embedding_model.embed_query(query)
    numpy_arr = np.array(query_embed).reshape(1, -1)

    distances, indices = retrieve_raw(query, embedding_model, vectorstore, k=3)

    for i in range(len(indices[0])):
        doc_id = vectorstore.index_to_docstore_id[indices[0][i]]
        doc = vectorstore.docstore.search(doc_id)
        print(f"Retrieved document {i+1} content:", doc.page_content)
        print("------")


    print("Retrieved documents with retrieve:")

    results = retrieve(query, vectorstore, k=3)
    for i, result in enumerate(results):
        print(f"Retrieved document {i+1} content:", result.page_content)
        print("------")