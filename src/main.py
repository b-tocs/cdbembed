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
        return context.create_success_messaga()
    else:
        return context.create_error_messaga()

class LearnModelInput(BaseModel):
    type: str = "default"
    name: str = "default"
    parameters: dict = {}

@app.post("/load_model")
async def loaded_model(data: LearnModelInput):
    context = Factory.new_context()
    handler = Factory.get_service_handler()
    if handler.load_model(context, data.type, data.name):
        return context.create_success_messaga()
    else:
        return context.create_error_messaga()


# ======================= StartUp
if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")