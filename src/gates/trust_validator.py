from transformers import pipeline
import torch

class TrustValidator:
    def __init__(self, model_name):
        self.pipe = pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
    
    def evaluate(self, context, answer):
        # Checks for Entailment vs Contradiction
        input_text = f"{context} [SEP] {answer}"
        result = self.pipe(input_text)[0]
        # Return a higher score for 'entailment'
        return 1.0 if result['label'] == 'entailment' else 0.0
