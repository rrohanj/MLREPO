import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class TopKKLMetric:
    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto" if torch.cuda.is_available() else None, 
            quantization_config=bnb_config if torch.cuda.is_available() else None
        )

    def calculate_shift(self, query, context, answer):
        # FIX: Ensure it doesn't crash if CUDA isn't detected
        inputs = self.tokenizer(f"Context: {context} Answer: {answer}", return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return loss.item()