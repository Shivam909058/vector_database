import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class VectorDatabase:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.database = {}

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def add(self, key, text):
        embedding = self.get_embedding(text)[0]
        self.database[key] = (text, embedding)

    def search(self, query):
        query_embedding = self.get_embedding(query)[0].reshape(1, -1)
        keys = list(self.database.keys())
        embeddings = np.array([value[1] for value in self.database.values()])

        similarities = cosine_similarity(query_embedding, embeddings)
        best_match_idx = np.argmax(similarities)
        best_match_key = keys[best_match_idx]

        return best_match_key, similarities[0][best_match_idx]

    def view_all(self):
        return self.database

    def delete(self, key):
        if key in self.database:
            del self.database[key]

    def clear(self):
        self.database.clear()
