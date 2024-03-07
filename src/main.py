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


@app.get("/loaded_models")
async def get_loaded_models():
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.get_loaded_models(context):
        return context.create_success_message()
    else:
        return context.create_error_message()

class LoadModelInput(BaseModel):
    type: str = "default"
    name: str = "default"
    id: str = None
    parameters: dict = {}

@app.post("/load_model")
async def load_model(data: LoadModelInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.load_model(context=context, model_type=data.type, model_name=data.name, model_id=data.id, parameters=data.parameters):
        return context.create_success_message()
    else:
        return context.create_error_message()

class UnloadModelInput(BaseModel):
    type: str = "default"
    name: str = "default"
    id: str = None

@app.post("/unload_model")
async def unload_model(data: UnloadModelInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.unload_model(context=context, model_type=data.type, model_name=data.name, model_id=data.id):
        return context.create_success_message()
    else:
        return context.create_error_message()
    
@app.post("/unload_all")
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