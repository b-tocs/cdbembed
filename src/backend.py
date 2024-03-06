from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction

class Context:
    def __init__(self, message: str = None, reason: str = None, status_code: int = 200) -> None:
        self.message: str = message
        self.reason: str = reason
        self.status_code: int = status_code

    def set_error(self, reason: str, status_code: int = 500):
        self.reason = reason
        self.status_code = status_code

    def set_success(self, message: str, reason: str = None, status_code: int = None):
        self.message = message
        self.reason  = reason

class EmbeddingFunctionInterface:
    
    def unload(self) -> bool:
        True

class EmbeddingFunctionDefault:

    def __init__(self, embedding_function: EmbeddingFunction) -> None:
        self.emedding_function = embedding_function


    


class ServiceHandler:
    VALID_TYPES = ["default"]

    def __init__(self) -> None:
        self._model_cache = {}


    def get_model_id(self, model_type: str, model_name: str, model_id: str = None) -> str:
        """generates a unique model id for internal cache

        :param model_type: type of the loader (default = 'default')
        :type model_type: str
        :param model_name: name of the model depending on loeader type
        :type model_name: str
        :param model_id: optional: given id for internal cache, defaults to None
        :type model_id: str, optional
        :return: identifier for model for the internal cache
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

    def load_model(self, context: Context, model_type: str, model_name: str, model_id: str = None) -> bool:
        try:
            # check model
            if model_type not in self.VALID_TYPES:
                context.set_error(f"invalid model type: {model_type}")
                return False
            
            # load model
            emb_function: EmbeddingFunctionInterface = None
            if model_type == "default":
                default_emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.emb_model)
                if default_emb:
                    emb_function = EmbeddingFunctionDefault(default_emb)
            
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