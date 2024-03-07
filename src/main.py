import uvicorn
from fastapi import FastAPI
from backend import Factory, ServiceHandler
from pydantic import BaseModel
from utils import getenv, getenv_as_int


# ======================= FastAPI Configuration
app = FastAPI()
Factory.get_service_handler().startup()


# ======================= FastAPI Methods
# -------------- Model
@app.get("/models", tags=["model"])
async def get_loaded_models():
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    result = handler.get_loaded_models(context)
    if result is None:
        return context.create_error_message()
    else: 
        return result
    
class LoadModelInput(BaseModel):
    type: str = "default"
    name: str = "default"
    id: str = None
    parameters: dict = {}

@app.post("/model_load", tags=["model"])
async def load_model(data: LoadModelInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.load_model(context=context, model_type=data.type, model_name=data.name, model_id=data.id, parameters=data.parameters):
        return context.create_success_message()
    else:
        return context.create_error_message()
    
class EmbedModelInput(BaseModel):
    text: str
    id: str = None
    type: str = "default"
    name: str = "default"


class UnloadModelInput(BaseModel):
    type: str = "default"
    name: str = "default"
    id: str = None

@app.post("/model_unload", tags=["model"])
async def unload_model(data: UnloadModelInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.unload_model(context=context, model_type=data.type, model_name=data.name, model_id=data.id):
        return context.create_success_message()
    else:
        return context.create_error_message()
    
@app.post("/models_unload_all", tags=["model"])
async def unload_all_models():
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.unload_all(context=context):
        return context.create_success_message()
    else:
        return context.create_error_message()    


# -------------- Embedding
@app.post("/embedding", tags=["embedding"])
async def get_embedding(data: EmbedModelInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.get_embedding(context=context, text=data.text, model_type=data.type, model_name=data.name, model_id=data.id):
        return context.create_success_message()
    else:
        return context.create_error_message()    

class EmbedModelOllamaInput(BaseModel):
    model: str
    prompt: str

@app.post("/embeddings", tags=["embedding"])
async def get_embedding_ollama(data: EmbedModelOllamaInput):
    """get embedding in ollama format

    :param data: _description_
    :type data: EmbedModelOllamaInput
    :return: embedding vector in ollama format
    :rtype: json
    """
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.get_embedding(context=context, model_type="", model_name="", text=data.prompt, model_id=data.model):
        payload = context.payload
        context.set_payload({"embedding": payload})
        return context.create_success_message()
    else:
        return context.create_error_message()    


# -------------- Documents vectordb
@app.get("/documents_count", tags=["vectordb"])
async def count_documents():
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    result = handler.documemts_count(context)
    if result is None:
        return context.create_error_message()
    else: 
        return result


class DocumentUpsertInput(BaseModel):
    id: str
    document: str
    uri: str = None
    metadata: dict = {}
    embedding: list = None

@app.post("/document_learn", tags=["vectordb"])
async def learn_document(data: DocumentUpsertInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    result = handler.document_learn(context=context, id=data.id, document=data.document, embedding=data.embedding, uri=data.uri, metatdata=data.metadata)
    if result is None:
        return context.create_error_message()
    else: 
        return result

class DocumentQueryInput(BaseModel):
    max_records: int = 5
    document: str
    embedding: list = None
    metadata: dict = {}

@app.post("/document_query", tags=["vectordb"])
async def query_document(data: DocumentQueryInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    result = handler.documents_query(context=context, max_records=data.max_records, document=data.document, embedding=data.embedding, metadata=data.metadata)
    if result is None:
        return context.create_error_message()
    else: 
        return result

# ======================= StartUp
if __name__ == "__main__":
    uvicorn.run(app, port=getenv_as_int("REST_API_PORT", 8000), host=getenv("REST_API_HOST", "0.0.0.0"))