import time
import numpy as np
import retico_core
from retico_core.audio import AudioIU
from retico_core.text import TextIU

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

from queue import Queue, Empty, Full

import logging
from console_colors import ConsoleColors
from collections import deque


class ASR(retico_core.AbstractModule):
    """ASR Module"""

    @staticmethod
    def name():
        return "Automatic Speech Recognition Module"

    @staticmethod
    def description():
        return "A module that converts speech to text using a pre-trained ASR model."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.device = device
        self.model_id = model_id
        self.model = None
        self.buffer = deque()  # Buffer to store audio chunks received from VAD module

    def setup(self):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
            model_kwargs={"language": "en"},
        )
        logging.info(f"{ConsoleColors.BLUE}ASR:{ConsoleColors.RESET} Module setup done")

    def process_update(self, update_message):

        iu, ut = next(update_message)  # Update message contains only one IU

        if ut == retico_core.UpdateType.ADD:
            self.buffer.append(iu.raw_audio)
            logging.debug(
                f"{ConsoleColors.MAGENTA}ASR:{ConsoleColors.RESET} Received new audio chunk, buffer size = {len(self.buffer)}"
            )
            return None
        elif ut == retico_core.UpdateType.COMMIT:
            # COMMIT Audio => End of Turn chunck of audio
            # Update the buffer with the last chunk of audio
            self.buffer.append(iu.raw_audio)
            # Convert to list of bytes
            audio_np = np.frombuffer(b"".join(self.buffer), dtype=np.int16)
            # ASR on the audio
            start_time = time.time()
            result = self.pipe(audio_np)
            end_time = time.time()

            # Empty the buffer
            self.buffer.clear()
            logging.info(
                f"{ConsoleColors.BLUE}ASR:{ConsoleColors.RESET}, End of Turn, Inference Time = {end_time-start_time}(s), User speech {result['text']}"
            )
            # Create a new IU with the result
            output_iu = self.create_iu()
            output_iu.set_text(result["text"])
            return retico_core.UpdateMessage.from_iu(
                output_iu, retico_core.UpdateType.COMMIT
            )
        elif ut == retico_core.UpdateType.REVOKE:  # Case False Turn Start
            # Empty the buffer
            self.buffer.clear()
            logging.debug(
                f"{ConsoleColors.MAGENTA}ASR:{ConsoleColors.RESET} Received REMOVE, clearing buffer"
            )
            return None
