from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = Path(__file__).resolve().parents[1] / "index" / "faiss_bge_m3"
EMB_NAME = "BAAI/bge-m3"

emb = HuggingFaceEmbeddings(
    model_name=EMB_NAME,
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)

def search(q: str, k: int = 5):
    docs = db.similarity_search(q, k=k)
    print(f"\nQUERY: {q}\n")
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        print(f"[{i}] {meta.get('title')}  |  {meta.get('source')}  |  {meta.get('chunk_id')}")
        print(d.page_content[:400].strip(), "...\n")

if __name__ == "__main__":
    search("Как называется первая серия первого сезона сериала Хворостов?", k=2)
    search("КГде находится Переулок Бестужева 221В ?", k=2)
    search("Краткое содержание Последнее дело", k=2)
