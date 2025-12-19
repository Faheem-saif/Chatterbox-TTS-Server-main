# File: server.py
# Main FastAPI application for the TTS Server with MULTILINGUAL SUPPORT.

import os
import io
import logging
import logging.handlers
import shutil
import time
import uuid
import yaml
import numpy as np
import librosa
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Literal
import webbrowser
import threading

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- Internal Project Imports ---
from config import (
    config_manager,
    get_host,
    get_port,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_language,  # ← Used for default language
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)

import engine  # TTS Engine (now supports multilingual)
from models import (
    CustomTTSRequest,
    ErrorResponse,
    UpdateStatusResponse,
)
import utils

from pydantic import BaseModel, Field


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"
    speed: float = 1.0
    seed: Optional[int] = None


# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Global Variables & Application Setup ---
startup_complete_event = threading.Event()


def _delayed_browser_open(host: str, port: int):
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning("Startup timeout. Browser not opened.")
            return
        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Opening browser: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TTS Server: Initializing...")
    try:
        paths_to_ensure = [
            get_output_path(),
            get_reference_audio_path(),
            get_predefined_voices_path(),
            Path("ui"),
            config_manager.get_path("paths.model_cache", "./model_cache", ensure_absolute=True),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        if not engine.load_model():
            logger.critical("CRITICAL: TTS Model failed to load.")
        else:
            logger.info("TTS Model loaded successfully.")
            host_address = get_host()
            server_port = get_port()
            browser_thread = threading.Thread(
                target=lambda: _delayed_browser_open(host_address, server_port),
                daemon=True,
            )
            browser_thread.start()

        logger.info("Startup complete.")
        startup_complete_event.set()
        yield
    finally:
        logger.info("TTS Server: Shutdown complete.")


app = FastAPI(
    title=get_ui_title(),
    description="Multilingual Text-to-Speech server with advanced UI and API.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
    if (ui_static_path / "vendor").is_dir():
        app.mount("/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files")

@app.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404)

@app.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404)

outputs_static_path = get_output_path(ensure_absolute=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_static_path)), name="generated_outputs")

templates = Jinja2Templates(directory=str(ui_static_path))

# --- API Endpoints (unchanged until /tts) ---

@app.get("/", response_class=HTMLResponse)
async def get_web_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/model-info")
async def get_model_info_endpoint():
    return engine.get_model_info()

@app.get("/api/ui/initial-data")
async def get_ui_initial_data():
    # ... (unchanged)
    pass  # Keep original logic

# ... (save_settings, reset_settings, restart_server, upload endpoints unchanged)

# --- MAIN TTS ENDPOINT WITH MULTILINGUAL SUPPORT ---
@app.post("/tts")
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
):
    if not engine.MODEL_LOADED:
        raise HTTPException(status_code=503, detail="TTS engine not loaded.")

    logger.info(f"TTS request: mode='{request.voice_mode}', format='{request.output_format}'")

    # --- Resolve voice prompt path ---
    audio_prompt_path_for_engine: Optional[Path] = None
    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(status_code=400, detail="Missing predefined_voice_id")
        potential_path = get_predefined_voices_path(ensure_absolute=True) / request.predefined_voice_id
        if not potential_path.is_file():
            raise HTTPException(status_code=404, detail="Predefined voice not found")
        audio_prompt_path_for_engine = potential_path

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(status_code=400, detail="Missing reference_audio_filename")
        potential_path = get_reference_audio_path(ensure_absolute=True) / request.reference_audio_filename
        if not potential_path.is_file():
            raise HTTPException(status_code=404, detail="Reference audio not found")
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        is_valid, msg = utils.validate_reference_audio(potential_path, max_dur)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid reference: {msg}")
        audio_prompt_path_for_engine = potential_path

    # --- Determine language_id (critical for multilingual) ---
    language_id = (
        request.language
        if request.language is not None
        else config_manager.get_string("generation_defaults.language", "en")
    )
    logger.info(f"Using language_id: {language_id}")

    # --- Chunking ---
    if request.split_text and len(request.text) > 180:
        chunk_size = request.chunk_size or 300
        text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size)
    else:
        text_chunks = [request.text]

    all_audio_segments_np: List[np.ndarray] = []
    engine_output_sample_rate: Optional[int] = None

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}")

        chunk_audio_tensor, chunk_sr = engine.synthesize(
            text=chunk,
            audio_prompt_path=str(audio_prompt_path_for_engine) if audio_prompt_path_for_engine else None,
            temperature=request.temperature or get_gen_default_temperature(),
            exaggeration=request.exaggeration or get_gen_default_exaggeration(),
            cfg_weight=request.cfg_weight or get_gen_default_cfg_weight(),
            seed=request.seed if request.seed is not None else get_gen_default_seed(),
            language_id=language_id,  # ← THIS ENABLES SPANISH AND OTHER LANGUAGES
        )

        if chunk_audio_tensor is None:
            raise HTTPException(status_code=500, detail=f"Failed to synthesize chunk {i+1}")

        if engine_output_sample_rate is None:
            engine_output_sample_rate = chunk_sr

        # Speed factor
        speed = request.speed_factor or get_gen_default_speed_factor()
        if speed != 1.0:
            chunk_audio_tensor, _ = utils.apply_speed_factor(chunk_audio_tensor, chunk_sr, speed)

        processed_np = chunk_audio_tensor.cpu().numpy().squeeze()
        all_audio_segments_np.append(processed_np)

    # --- Stitching & Encoding (unchanged from your smart stitching version) ---
    # (Keep all your excellent crossfade/smart stitching code here - it's already perfect)

    # Example (your existing stitching logic goes here)
    final_audio_np = np.concatenate(all_audio_segments_np) if len(all_audio_segments_np) > 1 else all_audio_segments_np[0]

    output_format_str = request.output_format or get_audio_output_format()
    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=get_audio_sample_rate(),
    )

    if not encoded_audio_bytes:
        raise HTTPException(status_code=500, detail="Audio encoding failed")

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    download_filename = utils.sanitize_filename(f"tts_{timestamp_str}.{output_format_str}")
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    return StreamingResponse(io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers)


# --- OpenAI Endpoint (optional language support) ---
@app.post("/v1/audio/speech")
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    # ... (keep existing, optionally add language if needed)
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=get_host(), port=get_port(), log_level="info")