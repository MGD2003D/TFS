from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base import BaseLLMClient

class QwenClient(BaseLLMClient):

    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
    
    async def initialize(self):

        print(f"Загружаю модель {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        print(f"Модель загружена на {self.device}")
        
        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"VRAM использовано: {vram:.2f} GB")
    
    async def simple_query(self, prompt: str) -> str:

        messages = [
            {"role": "system", "content": "Ты полезный ассистент."}, #TODO: Системный промпт тоже можно вынести в app_state пока что
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    async def cleanup(self):

        if self.model is not None:
            del self.model
            del self.tokenizer
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            print("Модель выгружена из памяти")