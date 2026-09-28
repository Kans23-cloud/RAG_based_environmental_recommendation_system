import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Load the dataset
df = pd.read_csv('data/dataset.csv')

# Create documents from the dataset
documents = []
metadatas = []
ids = []

for idx, row in df.iterrows():
    doc = f"Activity: {row['Activity']}. Average CO2 Emission: {row['Avg_CO2_Emission(kg/day)']} kg/day. Category: {row['Category']}. This activity contributes to CO2 emissions in the {row['Category']} category."
    documents.append(doc)
    metadatas.append({"activity": row['Activity'], "emission": row['Avg_CO2_Emission(kg/day)'], "category": row['Category']})
    ids.append(f"doc_{idx}")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(documents)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Create or get collection
collection = client.get_or_create_collection(name="co2_activities")

# Add to collection
collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=ids
)

print("Vector store created successfully!")
