from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction
from interfaces import EmbeddingFunctionInterface
from utils import Context


class EmbeddingFunctionDefault(EmbeddingFunctionInterface):

    DEFAULT_EMB_MODEL   = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "default", embedding_function: EmbeddingFunction = None, parameters: dict = {}) -> None:
        super().__init__(type_desc="ChromaDB sentence transformer", model_name=model_name, parameters=parameters)
        if model_name == "default":
            self.model_name = self.DEFAULT_EMB_MODEL

        self.emedding_function: EmbeddingFunction = embedding_function
    
    def load(self, context: Context) -> bool:
        if not self.model_name:
            return False
        try:
            self.emedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)
            if self.emedding_function:
                return True
            else:
                context.set_error(reason=f"Loading model {self.model_name} failed")
                return False
        except Exception as exc:
            context.set_error(reason=f"Loading model {self.model_name} failed - {exc}", status_code=500)
            return False
        
    
    def unload(self) -> bool:
        self.emedding_function = None
        return True
    
    def get_embedding(self, context: Context, text: str) -> list:
        if not self.emedding_function:
            context.set_error(f"embedding function not available for {self.get_description()}")
            return None
        else:
            try:
                result = self.emedding_function([text])
                if result and isinstance(result, list) and len(result) == 1:
                    context.set_payload(result[0])
                    return result[0]
                else:
                    context.set_error("invalid embedding")
                    return None            
            except Exception as exc:
                context.set_error(f"creating embedding failed: {exc}")
                return None
