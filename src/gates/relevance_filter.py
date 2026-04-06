from sentence_transformers import SentenceTransformer, util

class RelevanceFilter:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def evaluate(self, query, context):
        embeddings = self.model.encode([query, context], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1])
        return score.item()
