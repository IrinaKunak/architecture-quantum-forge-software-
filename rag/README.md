# Векторный индекс базы знаний (FAISS + bge-m3)

- Модель эмбеддингов: BAAI/bge-m3 (Sentence-Transformers)
- Ссылки: https://huggingface.co/BAAI/bge-m3
- Размер эмбеддингов: 1024
- База знаний: 34 файла в папке /data
- Разбиение: RecursiveCharacterTextSplitter (chunk_size=1200, overlap=200)
- Метаданные чанков: {source, title, chunk_id}
- Векторная БД: FAISS (index/faiss_bge_m3)
- Результаты:
  - Чанков в индексе: 200
  - Время генерации: 4 минуты
- Скрипт для сборки python scripts/build_index.py
- Скрипт для поиска python scripts/query_index.py
