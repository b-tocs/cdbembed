from utils import Context
class EmbeddingFunctionInterface:
    def __init__(self, type_desc: str, model_name: str, model_desc: str = None, parameters: dict = {}) -> None:
        self.model_name = model_name
        self.model_desc = model_desc
        self.type_desc  = type_desc
        self.parameters: dict = {}
        if not self.model_desc:
            self.model_desc = self.model_name

    def get_description(self) -> str:
        return f"Embedding Function type '{self.type_desc}' model '{self.model_desc}'"    
    
    def unload(self) -> bool:
        return True

    def load(self, context: Context) -> bool:
        context.set_error("abstract interface used")
        return False
    
    def get_embedding(self, context: Context, text: str) -> list:
        context.set_error("abstract interface used")
        return None
