import torch
from TTS.api import TTS

from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig

add_safe_globals([
    XttsConfig
])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Clone voice
tts.tts_to_file(
    text="This is my cloned voice speaking.",
    speaker_wav="audios/default.wav",
    language="en",
    file_path="audios/cloned_output.wav"
)


