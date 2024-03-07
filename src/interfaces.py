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


class VectorDBInterface:
    def __init__(self, host: str = None, port: int = None, url: str = None, collection: str = "default", parameters: dict = {}) -> None:
        self.host: str = host
        self.port: int = port
        self.url: str = url
        self.collection: str = collection
        self.parameters: dict = parameters
        
    def is_valid(self) -> bool:
        return False
    
    def learn_document(self, context: Context, id: str, document: str = None, embedding: list = None, uri: str = None, metadata: dict = {}) -> bool:
        return False
    
    def query_document(self, context: Context, max_records: int = 5, embedding: list = None, metadata: dict = None) -> bool:
        return False
    
    def count(self, context: Context) -> bool:
        return False
    
    def get_embedding_name(self) -> str:
        return self.parameters.get("embedding", "default")