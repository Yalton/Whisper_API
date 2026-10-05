from contextlib import asynccontextmanager, suppress
import asyncio

from fastapi import FastAPI
from modules.routes import router as transcription_router
from modules.transcription import idle_eviction_loop, purge_upload_directory
import uvicorn


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Single process: nothing in flight yet, so anything left is from a
    # previous process that was killed mid-request.
    purge_upload_directory()
    task = asyncio.create_task(idle_eviction_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(lifespan=lifespan)
app.include_router(transcription_router)
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
