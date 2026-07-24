from fastapi import FastAPI
from modules.routes import router as transcription_router
import uvicorn

app = FastAPI()
app.include_router(transcription_router)
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")