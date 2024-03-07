from interfaces import EmbeddingFunctionInterface
from utils import Context, getenv
import requests
import json


class EmbeddingFunctionOllama(EmbeddingFunctionInterface):

    DEFAULT_EMB_MODEL   = "llama2"

    def __init__(self, model_name: str = "default", parameters: dict = {}) -> None:
        super().__init__(type_desc="Ollama Embedding Proxy", model_name=model_name, parameters=parameters)
        self.url: str = parameters.get("url", None)
        if model_name == "default":
            self.model_name = self.DEFAULT_EMB_MODEL
    
    def load(self, context: Context) -> bool:        
        if not self.model_name or not self.url:
            context.set_error("model name or url in params is missing")
            return False
        else:
            return True
        
    
    def unload(self) -> bool:
        return True
    
    def get_embedding(self, context: Context, text: str) -> list:
        if not self.url:
            context.set_error("invalid route to ollama")
            return False
        try:
            url = f"{self.url}/api/embeddings"
            payload = {"model":self.model_name, "prompt":text}
            response = requests.post(url=url, json=payload)
            if not response.ok:
                context.set_error(f"call to ollama failed: {response.status_code} - {response.reason}")
                return None
            else:
                result = response.json()
                embedding = result.get("embedding")
                if embedding:
                    return embedding
                else:
                    context.set_error(f"invalid ollama embedding")
                    return None
        except Exception as exc:
            context.set_error(f"calling ollama failed: {exc}")
            return None
        
