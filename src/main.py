import uvicorn
from fastapi import FastAPI
from .backend import Factory, ServiceHandler



# ======================= FastAPI Configuration
app = FastAPI()


# ======================= FastAPI Methods
@app.get("/info")
async def root():
    return {"message": "Hello World"}



# ======================= StartUp
if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")