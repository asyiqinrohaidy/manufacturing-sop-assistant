from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import upload, ask

app = FastAPI(title="Manufacturing SOP Assistant")

# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(ask.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Manufacturing SOP Assistant API is running"}