import threading
import time
import numpy as np
import retico_core
from retico_core.audio import AudioIU
from retico_core.text import TextIU
from kokoro import KPipeline
import soundfile as sf
import torchaudio
import torch
import logging
from console_colors import ConsoleColors

import queue


class TTS(retico_core.AbstractModule):
    """TTS Module
    TTS Module runs two thereads :
        1- One to receive update messages and add them to a queue
        2- One that
    """

    @staticmethod
    def name():
        return "Text To Speech Module"

    @staticmethod
    def description():
        return "A module that converts system utterance to a speech."

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(
        self, sample_rate, model_args, output_audio_bytes=True, sample_width=2, **kwargs
    ):
        super().__init__(**kwargs)
        self.buffer_in = queue.Queue()
        self.sample_rate = sample_rate
        self.model_args = model_args
        self.producer = None

        # Params  for Audio output : bytes or tensor
        self.output_audio_bytes = output_audio_bytes
        self.sample_width = sample_width

    def setup(self):
        self.producer = KoKoRoTTS(
            buffer_in=self.buffer_in,
            sample_rate=self.sample_rate,
            callback=self.send_message,
            model_args=self.model_args,
        )

    def prepare_run(self):
        if self.producer is not None:
            self.producer.start()
        super().prepare_run()

    def process_update(self, update_message):
        iu, ut = next(update_message)  # TTS returns the transcript of one speech
        # TODO : Use streaming with .ADD and .COMMIT which uses either a queue or streaming tts model
        if ut == retico_core.UpdateType.COMMIT:
            logging.debug(
                f"{ConsoleColors.MAGENTA}TTS:{ConsoleColors.RESET}: Received a COMMIT Message"
            )
            self.buffer_in.put(iu.text)

    def send_message(self, raw_audio, is_final: bool = False):
        """Method to create an output message"""
        output_iu = self.create_iu()

        nframes = len(raw_audio)

        # Convert the audio to bytes if next modules takes bytes as input
        if self.output_audio_bytes:
            raw_audio = torch.clamp(raw_audio, -1.0, 1.0)
            audio_int16 = (raw_audio * 32767.0).to(torch.int16)

            # Convert to bytes
            audio = audio_int16.numpy().tobytes()
            nframes = audio_int16.shape[0]
        else:
            audio = raw_audio.numpy()

        output_iu.set_audio(
            raw_audio=audio,
            nframes=nframes,
            rate=self.sample_rate,
            sample_width=self.sample_width,
        )
        if is_final:
            out_message = retico_core.UpdateMessage.from_iu(
                output_iu, retico_core.UpdateType.COMMIT
            )
        else:
            out_message = retico_core.UpdateMessage.from_iu(
                output_iu, retico_core.UpdateType.ADD
            )
        self.append(out_message)
        logging.debug(
            f"{ConsoleColors.MAGENTA}TTS:{ConsoleColors.RESET} Sending a message, is_final = {is_final}"
        )

    def shutdown(self):
        # Add the model producer setting to end
        if self.producer is not None:
            self.producer.stop()
        super().shutdown()


class KoKoRoTTS(threading.Thread):
    """Class for TTS: retrieve text from queue and generate the corresponding speech"""

    def __init__(
        self, buffer_in, sample_rate, callback, model_args, voice: str = "am_fenrir"
    ):
        super().__init__()
        self.buffer_in = buffer_in
        self.callback = callback  # Method to call at the end of the generative process
        self._stop_event = threading.Event()
        self.sample_rate = sample_rate
        self.pipeline = KPipeline(**model_args)
        self.resampler = None
        if self.sample_rate != 24000:  # Base freq for KoKoRo TTS
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=24000, new_freq=self.sample_rate
            )
        self.voice = voice

    def run(self):
        while not self._stop_event.is_set():
            try:
                text = self.buffer_in.get(timeout=1)

                # Generate speech
                start_time = time.time()
                generator = self.pipeline(text, voice=self.voice)

                # Send sapeech
                last_item = None
                nb_chuncks = 0
                try:
                    while True:
                        item = next(generator)
                        if last_item is not None:
                            gs, ps, audio = last_item
                            if self.resampler is not None:
                                audio = self.resampler(audio)
                            self.callback(audio, False)  # Not final audio chunck
                            nb_chuncks += 1
                        last_item = item
                except StopIteration:  # Last Audio Chunck
                    if last_item is not None:
                        gs, ps, audio = last_item
                        if self.resampler is not None:
                            audio = self.resampler(audio)
                        self.callback(audio, True)  # Not final audio chunck
                        nb_chuncks += 1

                end_time = time.time()
                logging.info(
                    f"{ConsoleColors.BLUE}KoKoRoTTS:{ConsoleColors.RESET} End of Inference , delay = {end_time - start_time} (s), Nb audio chuncks : {nb_chuncks}"
                )
            except queue.Empty:  # To check for stop_event flag
                continue

    def stop(self):
        """Set the stop flag"""
        self._stop_event.set()
