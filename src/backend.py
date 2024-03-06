from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction
from fastapi.responses import HTMLResponse 
import json

class Context:
    def __init__(self, message: str = None, reason: str = None, status_code: int = None) -> None:
        self.message: str = message
        self.reason: str = reason
        self.status_code: int = status_code


    def set_error(self, reason: str, status_code: int = 400):
        self.reason = reason
        self.status_code = status_code

    def set_success(self, message: str, reason: str = None, status_code: int = None):
        self.message = message
        self.reason  = reason
        if status_code:
            self.status_code = status_code

    def create_success_messaga(self):
        message = self.message
        if not message:
            message = 'OK'
        return {
            "message": message
        }    

    def create_error_messaga(self):
        reason = "unnown error"
        if self.reason:
            reason = self.reason

        message = self.message
        if not message:
            message = self.reason
        
        status = 400
        if self.status_code:
            status = self.status_code

        payload = json.dumps({
            "message": message
        })    

        return HTMLResponse(content=payload, status_code=status)


class EmbeddingFunctionInterface:
    def __init__(self, type_desc: str, model_desc: str) -> None:
        self.model_text = model_desc
        self.type_text  = type_desc

    def get_description(self) -> str:
        return f"Embedding Function type '{self.type_text}' model '{self.model_text}'"    
    
    def unload(self) -> bool:
        True

class EmbeddingFunctionDefault(EmbeddingFunctionInterface):

    DEFAULT_EMB_MODEL   = "all-MiniLM-L6-v2"

    def __init__(self, embedding_function: EmbeddingFunction, model_name: str = "default") -> None:
        super().__init__(type_desc="ChromaDB sentence transformer", model_desc=model_name)
        if model_name == "default":
            self.model_text = self.DEFAULT_EMB_MODEL

        self.emedding_function = embedding_function
    


class ServiceHandler:
    VALID_TYPES = ["default"]

    def __init__(self) -> None:
        self._model_cache:dict[str, EmbeddingFunctionInterface] = {}


    def get_model_id(self, model_type: str, model_name: str, model_id: str = None) -> str:
        """generates a unique model id for internal cache

        :param model_type: type of the loader (default = 'default')
        :type model_type: str
        :param model_name: name of the model depending on loeader type
        :type model_name: str
        :param model_id: optional: given id for internal cache, defaults to None
        :type model_id: str, optional
        :return: identifier for model in the internal cache
        :rtype: str
        """
        if model_id:
            return model_id
        elif model_type == "default":
            return model_name 
        else:
            return f"{model_type}::{model_name}"

    def is_model_loaded(self, context: Context, model_type: str, model_name: str, model_id: str = None) -> bool:
        use_model_id = self.get_model_id(model_type=model_type, model_name=model_name, model_id=model_id)
        if use_model_id in self._model_cache:
            return True
        else:
            return False
        
    def get_loaded_models(self, context: Context) -> list:
        result = []
        for model_id in self._model_cache.keys():
            emb_function = self._model_cache.get(model_id)
            record = {
                "id": model_id,
                "description": emb_function.get_description()
            }   
            result.append(record)
        return result

    def get_embedding_function_by_id(self, model_id: str) -> EmbeddingFunctionInterface:
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        else:
            return None

    def unload_model(self, context: Context, model_type: str, model_name: str, model_id: str = None) -> bool:
        use_model_id = self.get_model_id(model_type=model_type, model_name=model_name, model_id=model_id)
        if not use_model_id in self._model_cache:
            context.set_error("model not loaded", status_code=400)
            return False
        else:
            emb_func = self.get_embedding_function_by_id(use_model_id)
            emb_func.unload()
            del self._model_cache[use_model_id]
            return True

    def unload_all(self, context: Context) -> True:
        count = 0
        for model_id in self._model_cache.keys():
            self.unload_model(context, "", "", model_id=model_id)
            count += 1
        context.set_success(f"{count} models unloaded")
        return True

    def load_model(self, context: Context, model_type: str, model_name: str, model_id: str = None, parameters: dict = {}) -> bool:
        try:
            # check model
            if model_type not in self.VALID_TYPES:
                context.set_error(f"invalid model type: {model_type}")
                return False
            
            # load model
            emb_function: EmbeddingFunctionInterface = None
            if model_type == "default":
                default_emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
                if default_emb:
                    emb_function = EmbeddingFunctionDefault(embedding_function=default_emb, model_name=model_name)
            
            # check 
            if not emb_function:
                context.set_error(f"invalid embedding functin: loader type {model_type} model name {model_name}")
                return False

            # get the model id
            use_model_id = self.get_model_id(model_type=model_type, model_name=model_name, model_id=model_id)
            self._model_cache[use_model_id] = emb_function
            return True

        except Exception as exc:
            context.set_error(f"Error: {exc}")



class Factory:

    _service_handler = ServiceHandler()

    @classmethod
    def get_service_handler(cls) -> ServiceHandler:
        return cls._service_handler
    
    @classmethod
    def new_context(cls) -> Context:
        return Context()