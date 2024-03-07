import uvicorn
from fastapi import FastAPI
from backend import Factory, ServiceHandler
from pydantic import BaseModel


# ======================= FastAPI Configuration
app = FastAPI()


# ======================= FastAPI Methods
@app.get("/info")
async def root():
    return {"message": "Hello World"}


@app.get("/models")
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

@app.post("/model_load")
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


@app.post("/embedding")
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

@app.post("/embeddings")
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




class UnloadModelInput(BaseModel):
    type: str = "default"
    name: str = "default"
    id: str = None

@app.post("/model_unload")
async def unload_model(data: UnloadModelInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.unload_model(context=context, model_type=data.type, model_name=data.name, model_id=data.id):
        return context.create_success_message()
    else:
        return context.create_error_message()
    
@app.post("/models_unload_all")
async def unload_all_models():
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.unload_all(context=context):
        return context.create_success_message()
    else:
        return context.create_error_message()    

# ======================= StartUp
if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")