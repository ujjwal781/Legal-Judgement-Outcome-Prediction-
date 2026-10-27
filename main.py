
#  IMPORTS

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# LOAD LOCAL CSV FILES

print("Loading dataset from CSV...")

dataset = load_dataset(
    "csv",
    data_files={
        "train": "train_multi.csv",
        "test": "test.csv",
        "validation": "dev.csv"
    }
)

print(dataset)


#CHECK STRUCTURE

print("\nSample row:\n", dataset["train"][0])


# PREPARE TEXT

print("\nPreparing text...")

#  create subset FIRST
subset = dataset["train"].select(range(1000))

# create texts
texts = [
    f"Case: {item['Case Name']}\n\n{item['text']}"
    for item in subset
]


print(f"Total documents: {len(texts)}")


#  CHUNKING

print("\nChunking text...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

documents = text_splitter.create_documents(texts)

print(f"Total chunks created: {len(documents)}")


# EMBEDDINGS

print("\nLoading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en",
    model_kwargs={"device": "cpu"},   
    encode_kwargs={"batch_size": 32}
)


#  CREATE VECTOR DB

print("\nCreating FAISS vector database (this will take time)...")

vector_db = FAISS.from_documents(documents, embedding_model)


# SAVE VECTOR DB

print("\nSaving FAISS index...")

vector_db.save_local("faiss_legal_index")

print("\n Vector DB created and saved successfully!")


# TEST QUERY

print("\nRunning test query...\n")

query = "violation of IBC moratorium"

results = vector_db.similarity_search(query, k=3)

for i, res in enumerate(results):
    print(f"\n🔹 Result {i+1}:")
    print(res.page_content[:500])