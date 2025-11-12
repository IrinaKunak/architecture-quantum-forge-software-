import os, json, hashlib
from pathlib import Path
from tqdm import tqdm
from slugify import slugify

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDEX_DIR = Path(__file__).resolve().parents[1] / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_documents():
    docs = []
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*",
        show_progress=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8", "autodetect_encoding":True},
        use_multithreading=True,
        silent_errors=True
    )
    docs += loader.load()
    return docs


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n## ", "\n# ", "\n", " ", ""]
)


EMB_NAME = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_NAME,
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def chunk_with_meta(docs):
    chunks = []
    for d in tqdm(docs, desc="Chunking"):

        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        rel = os.path.relpath(src, DATA_DIR) if src != "unknown" else "unknown"
        title = Path(src).stem.replace("_", " ") if src != "unknown" else "Untitled"

        text = d.page_content or ""

        for i, chunk in enumerate(splitter.split_text(text)):
            h = hashlib.sha1(f"{rel}#{i}".encode("utf-8")).hexdigest()[:12]
            chunks.append(type(d)(page_content=chunk, metadata={
                "source": rel,
                "title": title,
                "chunk_id": f"{slugify(title)}-{h}",
            }))
    return chunks

def main():
    print(f"Загружаем БЗ из {DATA_DIR} ...")
    docs = load_documents()
    print(f"Загружено {len(docs)} документов")

    chunks = chunk_with_meta(docs)
    print(f"Обработано {len(chunks)} чанков")

    print("Сделаем FAISS индекс...")
    vs = FAISS.from_documents(chunks, embeddings)

    vs.save_local(str(INDEX_DIR / "faiss_bge_m3"))
    info = {
        "embedding_model": EMB_NAME,
        "chunks_count": len(chunks),
        "files_count": len({c.metadata["source"] for c in chunks}),
    }
    (INDEX_DIR / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print("Готово. Индекс сохранен по пути index/faiss_bge_m3 и build_info.json")

if __name__ == "__main__":
    main()
