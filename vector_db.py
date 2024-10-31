import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorDB(nn.Module):
    
    def __init__(self, embed_model):
        super().__init__()
        self.embedding_model = SentenceTransformer(embed_model)

    def load_data(self, file_paths): 
        
        file_paths = list(file_paths)
        
        data =  ''
        
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                data += f.read()
        
        return data

    def chunk_data(self, data, chunk_size):
        
        assert isinstance(data, str)
        
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
    
    def index(self, chunks):
        
        chunk_embeddings = [self.embedding_model.encode(chunk) for chunk in chunks]
        return chunk_embeddings
    
    def retrieve_top_k(self, query, chunks, chunk_embeddings, k):
        
        query_vector = self.embedding_model.encode(query)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        scores = []
        
        for i, chunk_embedding in enumerate(chunk_embeddings):
        
            chunk_embedding = chunk_embedding / np.linalg.norm(chunk_embedding)
            score = np.dot(query_vector, chunk_embedding)
            scores.append((score, i))  
       
        scores.sort(reverse=True)
     
        results =  [(chunks[idx], score) for score, idx in scores[:k]]
        return results

db = VectorDB('all-MiniLM-L6-v2')
data = db.load_data('ML.txt')
chunks = db.chunk_data(data, 1000)
chunk_embeddings = db.index(chunks)

query = "What is machine learning?"
results = db.retrieve_top_k(query, chunks, chunk_embeddings, k=3)

for chunk, score in results:
    print(f"Score: {score:.4f}")
    print(f"Text: {chunk[:200]}")
    print("-" * 50)