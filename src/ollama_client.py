from interfaces import EmbeddingFunctionInterface
from utils import Context, getenv


class EmbeddingFunctionOllama(EmbeddingFunctionInterface):

    DEFAULT_EMB_MODEL   = "llama2"

    def __init__(self, model_name: str = "default", parameters: dict = {}) -> None:
        super().__init__(type_desc="Ollama Embedding Proxy", model_name=model_name, parameters=parameters)
        if model_name == "default":
            self.model_name = self.DEFAULT_EMB_MODEL
            self.url: str = parameters.get("url", None)
    
    def load(self, context: Context) -> bool:        
        if not self.model_name or not self.url:
            context.set_error("model name or url in params is missing")
            return False
        else:
            return True
        
    
    def unload(self) -> bool:
        return True
    
    def get_embedding(self, context: Context, text: str) -> list:
        return False