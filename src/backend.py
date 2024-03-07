from interfaces import EmbeddingFunctionInterface, VectorDBInterface
from utils import Context, getenv, getenv_as_int
from chroma import EmbeddingFunctionDefault
from ollama_client import EmbeddingFunctionOllama
from chroma_server import ChromaDBServer
    

class ServiceHandler:
    TYPE_DEFAULT = "default"
    TYPE_CHROMADB  = "chromadb"
    TYPE_OLLAMA  = "ollama"

    VALID_TYPES = [TYPE_DEFAULT, TYPE_OLLAMA]
    VECTORDB_TYPES = [TYPE_DEFAULT, TYPE_CHROMADB]

    def __init__(self) -> None:
        self._model_cache:dict[str, EmbeddingFunctionInterface] = {}
        self.vdb_server: VectorDBInterface = None

    def startup(self) -> bool:
        try:
            # check for default model
            context = Context()
            default_model = getenv("DEFAULT_MODEL")
            if default_model:
                print("Loading default model...")
                if self.load_model(context=context, model_type="default", model_name=default_model, model_id="default"):
                    print(f"Default model loaded at startup: {default_model}")
                else:
                    print(f"Loading default model {default_model} at startup failed")

            # check ollama
            ollama_url = getenv("OLLAMA_URL")
            ollama_model = getenv("OLLAMA_MODEL")
            if ollama_model and ollama_url:
                if self.load_model(context=context, model_type="ollama", model_name=ollama_model, model_id="ollama", parameters={"url": ollama_url}):
                    print(f"Ollama proxy loaded at startup: url {ollama_url} model {ollama_model}")
                else:
                    print(f"Loading ollama proxy to {ollama_url} at startup failed")      

            # check VectorDB
            vdb_type = getenv("VECTORDB_TYPE")                    
            vdb_host = getenv("VECTORDB_HOST")
            vdb_port = getenv_as_int("VECTORDB_PORT")
            vdb_coll = getenv("VECTORDB_COLLECTION", "default")
            vdb_emb  = getenv("VECTORDB_EMBEDDING", "default")
            vdb_dbs  = getenv("VECTORDB_DATABASE", "default_database")
            vdb_ten  = getenv("VECTORDB_TENANT","default_tenant")

            # init vdb istance
            vdb_server: VectorDBInterface = None
            if vdb_type:
                if vdb_type not in self.VECTORDB_TYPES:
                    print("invalid vectordb type - valid values: ", self.VECTORDB_TYPES)
                elif vdb_type == self.TYPE_DEFAULT or vdb_type == self.TYPE_CHROMADB:
                    if vdb_port:
                        vdb_server = ChromaDBServer(host=vdb_host, port=vdb_port, collection=vdb_coll, parameters={"database": vdb_dbs, "tenant": vdb_ten, "embedding": vdb_emb})
            
            if not vdb_server or not vdb_server.is_valid():
                print("connect to vector database failed")
                return False
            else:       
                self.vdb_server = vdb_server
                print("VectorDB connected.")     
                return True           
        except Exception as exc:
            print(f"Error while checking environment parameters at startup: {exc}")


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
        context.set_payload(result)
        return result

    def get_embedding_function_by_id(self, model_id: str) -> EmbeddingFunctionInterface:
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        else:
            return None

    def unload_model(self, context: Context, model_type: str, model_name: str, model_id: str = None, delete_cache: bool = True) -> bool:
        use_model_id = self.get_model_id(model_type=model_type, model_name=model_name, model_id=model_id)
        if not use_model_id in self._model_cache:
            context.set_error("model not loaded", status_code=400)
            return False
        else:
            emb_func = self.get_embedding_function_by_id(use_model_id)
            emb_func.unload()
            if delete_cache:
                del self._model_cache[use_model_id]
            return True

    def unload_all(self, context: Context) -> True:
        count = 0
        all_ids = self._model_cache.keys()
        for model_id in all_ids:
            self.unload_model(context, "", "", model_id=model_id, delete_cache=False)
            count += 1
        context.set_success(f"{count} models unloaded")
        self._model_cache = {}
        return True

    def load_model(self, context: Context, model_type: str, model_name: str, model_id: str = None, parameters: dict = {}) -> bool:
        try:
            # check model
            if model_type not in self.VALID_TYPES:
                context.set_error(f"invalid model type: {model_type}")
                return False
            
            # workaround swagger
            if model_id == "string":
                model_id = None

            # set embedding by type
            emb_function: EmbeddingFunctionInterface = None
            if model_type == "default":
                emb_function = EmbeddingFunctionDefault(model_name=model_name)
            elif model_type == "ollama":
                emb_function = EmbeddingFunctionOllama(model_name=model_name, parameters=parameters)
                
            # check and load model
            if not emb_function:
                context.set_error(f"invalid embedding function: loader type {model_type} model name {model_name}")
                return False
            else:
                if not emb_function.load(context=context):
                    context.set_error(reason="Loading model failed")
                    return False

            # get the model id
            use_model_id = self.get_model_id(model_type=model_type, model_name=model_name, model_id=model_id)
            self._model_cache[use_model_id] = emb_function
            context.set_success(f"model loaded as id {use_model_id}")
            return True

        except Exception as exc:
            context.set_error(f"Error: {exc}")

    def get_embedding(self, context: Context, text: str, model_type: str, model_name: str, model_id: str = None) -> bool:
        try:
            # get the model id and check if loaded
            use_model_id = self.get_model_id(model_type=model_type, model_name=model_name, model_id=model_id)
            if not self.is_model_loaded(context=context, model_name=model_name, model_type=model_type, model_id=use_model_id):
                if not self.load_model(context=context, model_type=model_id, model_name=model_name, model_id=model_id):
                    context.set_error("model not found")
                    return False

            # get embedding function
            emb_function = self.get_embedding_function_by_id(use_model_id)
            if not emb_function:
                context.set_error("model function not found")
                return False

            # get embedding
            result = emb_function.get_embedding(context=context, text=text)
            if result and isinstance(result, list) and len(result) > 0:
                context.set_payload(result)
                return True
            else:
                context.set_error("invalid embedding detected")
                return False

        except Exception as exc:
            context.set_error(f"Error: {exc}")
        
    def documents_query(self, context: Context, max_records: int = 5, document: str = None, embedding: list = None, metadata: dict = {}) -> bool:
        try:
            if not self.vdb_server:
                context.set_error("no vector engine connected")
                return False
            
            if not embedding and not document:
                context.set_error("document or embedding required")
                return False

            # check embedding
            if not embedding:
                emb_name = self.vdb_server.get_embedding_name()
                emb_func = self.get_embedding_function_by_id(emb_name)
                if not emb_func:
                    context.set_error(f"valid embedding required - {emb_name} invalid")
                    return False

                embedding = emb_func.get_embedding(context=context, text=document)
                if not embedding:
                    context.set_error(f"embedding genearation failed")
                    return False

            return self.vdb_server.query_document(context=context, max_records=max_records, embedding=embedding, metadata=metadata)

        except Exception as exc:
            context.set_error(f"Error: {exc}")
            return False
    
    def document_learn(self, context: Context, id: str, document: str = None, embedding: list = None, uri: str = None, metatdata: dict = {}) -> bool:
        try:
            # check
            if not self.vdb_server:
                context.set_error("no vector engine connected")
                return False

            if not id:
                context.set_error("id required")
                return False

            if not embedding and not document:
                context.set_error("document or embedding required")
                return False

            # check embedding
            if not embedding:
                emb_name = self.vdb_server.get_embedding_name()
                emb_func = self.get_embedding_function_by_id(emb_name)
                if not emb_func:
                    context.set_error(f"valid embedding required - {emb_name} invalid")
                    return False

                embedding = emb_func.get_embedding(context=context, text=document)
                if not embedding:
                    context.set_error(f"embedding genearation failed")
                    return False
                

            # learn with embedding
            return self.vdb_server.learn_document(context=context, id=id, document=document, embedding=embedding, uri=uri, metadata=metatdata)

        except Exception as exc:
            context.set_error(f"Error: {exc}")
            return False
    

    def documemts_count(self, context: Context) -> bool:
        try:
            if not self.vdb_server:
                context.set_error("no vector engine connected")
                return False

            return self.vdb_server.count(context)

        except Exception as exc:
            context.set_error(f"Error: {exc}")
            return False


class Factory:

    _service_handler = ServiceHandler()

    @classmethod
    def get_service_handler(cls) -> ServiceHandler:
        return cls._service_handler
    
    @classmethod
    def new_context(cls) -> Context:
        return Context()