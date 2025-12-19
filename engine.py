# File: engine.py
# Core TTS model loading and speech generation logic with MULTILINGUAL support

import gc
import logging
import random
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

# Standard imports
from chatterbox.tts import ChatterboxTTS  # Original English-focused model
from chatterbox.models.s3gen.const import S3GEN_SR

# Multilingual model import
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Defensive Turbo import
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    TURBO_AVAILABLE = True
except ImportError:
    ChatterboxTurboTTS = None
    TURBO_AVAILABLE = False

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# Log availability
if TURBO_AVAILABLE:
    logger.info("ChatterboxTurboTTS is available.")
else:
    logger.info("ChatterboxTurboTTS not available.")

# Model selector whitelist - NOW INCLUDES MULTILINGUAL
MODEL_SELECTOR_MAP = {
    # Original (English-focused)
    "chatterbox": "original",
    "original": "original",
    "resembleai/chatterbox": "original",

    # Turbo
    "chatterbox-turbo": "turbo",
    "turbo": "turbo",
    "resembleai/chatterbox-turbo": "turbo",

    # Multilingual (23 languages)
    "chatterbox-multilingual": "multilingual",
    "multilingual": "multilingual",
    "resembleai/chatterbox-multilingual": "multilingual",
}

# Paralinguistic tags (Turbo only)
TURBO_PARALINGUISTIC_TAGS = [
    "laugh", "chuckle", "sigh", "gasp", "cough",
    "clear throat", "sniff", "groan", "shush",
]

# --- Global Variables ---
chatterbox_model: Optional[any] = None  # Can be any of the three classes
MODEL_LOADED: bool = False
model_device: Optional[str] = None
loaded_model_type: Optional[str] = None  # "original", "turbo", or "multilingual"
loaded_model_class_name: Optional[str] = None


def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        test_tensor = torch.tensor([1.0]).cuda().cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        test_tensor = torch.tensor([1.0]).to("mps").cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS test failed: {e}")
        return False


def _get_model_class(selector: str) -> tuple:
    selector_normalized = selector.lower().strip()
    model_type = MODEL_SELECTOR_MAP.get(selector_normalized)

    if model_type == "multilingual":
        logger.info(f"Model selector '{selector}' resolved to Multilingual model (ChatterboxMultilingualTTS)")
        return ChatterboxMultilingualTTS, "multilingual"

    if model_type == "turbo":
        if not TURBO_AVAILABLE:
            raise ImportError("Turbo selected but not available. Update chatterbox-tts.")
        logger.info(f"Model selector '{selector}' resolved to Turbo model")
        return ChatterboxTurboTTS, "turbo"

    if model_type == "original":
        logger.info(f"Model selector '{selector}' resolved to Original model")
        return ChatterboxTTS, "original"

    # Default fallback
    logger.warning(f"Unknown selector '{selector}'. Defaulting to multilingual.")
    return ChatterboxMultilingualTTS, "multilingual"


def get_model_info() -> dict:
    return {
        "loaded": MODEL_LOADED,
        "type": loaded_model_type,
        "class_name": loaded_model_class_name,
        "device": model_device,
        "sample_rate": chatterbox_model.sr if chatterbox_model else None,
        "supports_paralinguistic_tags": loaded_model_type == "turbo",
        "available_paralinguistic_tags": TURBO_PARALINGUISTIC_TAGS if loaded_model_type == "turbo" else [],
        "turbo_available_in_package": TURBO_AVAILABLE,
        "supports_multilingual": loaded_model_type == "multilingual",
    }


def load_model() -> bool:
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name

    if MODEL_LOADED:
        logger.info("Model already loaded.")
        return True

    try:
        # Device selection (same as before)
        device_setting = config_manager.get_string("tts_engine.device", "auto")
        if device_setting == "auto":
            resolved_device_str = "cuda" if _test_cuda_functionality() else "mps" if _test_mps_functionality() else "cpu"
        elif device_setting in ["cuda", "mps", "cpu"]:
            resolved_device_str = device_setting if (device_setting != "cuda" or _test_cuda_functionality()) else "cpu"
            resolved_device_str = device_setting if (device_setting != "mps" or _test_mps_functionality()) else "cpu"
        else:
            resolved_device_str = "cuda" if _test_cuda_functionality() else "cpu"

        model_device = resolved_device_str
        logger.info(f"Using device: {model_device}")

        # Model selector - CHANGE THIS IN config.yaml TO "chatterbox-multilingual" FOR SPANISH!
        model_selector = config_manager.get_string("model.repo_id", "chatterbox-turbo")
        logger.info(f"Model selector: '{model_selector}'")

        model_class, model_type = _get_model_class(model_selector)

        logger.info(f"Loading {model_class.__name__} on {model_device}...")
        chatterbox_model = model_class.from_pretrained(device=model_device)

        loaded_model_type = model_type
        loaded_model_class_name = model_class.__name__

        MODEL_LOADED = True
        logger.info(f"Successfully loaded {model_class.__name__} (type: {model_type})")
        logger.info(f"Sample rate: {chatterbox_model.sr} Hz")

        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language_id: str = "en",  # NEW: Required for multilingual
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("Model not loaded.")
        return None, None

    try:
        if seed != 0:
            set_seed(seed)

        logger.debug(f"Synthesizing: lang={language_id}, text='{text[:50]}...'")

        # Multilingual model requires language_id
        if loaded_model_type == "multilingual":
            wav_tensor = chatterbox_model.generate(
                text=text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        else:
            # Original and Turbo don't use language_id
            wav_tensor = chatterbox_model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        return wav_tensor, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Synthesis error: {e}", exc_info=True)
        return None, None


def reload_model() -> bool:
    global chatterbox_model, MODEL_LOADED, loaded_model_type, loaded_model_class_name

    logger.info("Reloading model...")

    if chatterbox_model is not None:
        del chatterbox_model
        chatterbox_model = None

    MODEL_LOADED = False
    loaded_model_type = None
    loaded_model_class_name = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass

    return load_model()