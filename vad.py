# TODO: Implement a VAD module that works as a gate and forward the audio only if the user is speaking
# *  Detect audio activity, if not speaking ignore
# *      Otherwise , buffer the audio until the user stops speaking then forward to the next module
# *  End of turn if the speaker stops talking for 100ms
# * V01 : Direct model with no streaming mode
# * V02 : Streaming model with a buffer of 100ms
# *  In V02, start streaming audio to the next module with Commit when it reaches end of speech

from enum import Enum

import retico_core
from retico_core.audio import AudioIU, SpeechIU
import logging
import webrtcvad

from console_colors import ConsoleColors


class VADState(Enum):
    SILENCE = 1
    SPEECH = 2
    SILENCE_TURN = 3


class VAD(retico_core.AbstractModule):
    """VAD Module"""

    @staticmethod
    def name():
        return "Voice Activity Detection Module"

    @staticmethod
    def description():
        return "A module that detects voice activity in the audio stream and forwards the audio to the next module if voice activity is detected."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(
        self,
        mode=2,
        sample_rate=16000,
        frame_length=0.02,
        min_turn_length=0.1,
        max_silence_length=0.200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.min_turn_length = min_turn_length
        self.max_silence_length = max_silence_length

        self.speech_length = 0.0  # Total Length of speech detected
        self.silence_length = 0.0  # Total Length of silence detected
        self.state = VADState.SILENCE  # State of the module

    def setup(self):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.mode)
        logging.info(
            f"{ConsoleColors.BLUE} VAD:{self.state}:{ConsoleColors.RESET} Module setup done"
        )

    def process_update(self, update_message):
        iu, ut = next(update_message)
        # Receive one IU per message from the microphone module
        is_speech = self.vad.is_speech(iu.raw_audio, self.sample_rate)

        # Speech Detected, Forward the audio to the next module
        if is_speech:
            logging.debug(
                f"{ConsoleColors.MAGENTA}VAD:{self.state}:{ConsoleColors.RESET} Speech Detected, speech_length = {self.speech_length}, silence_length = {self.silence_length}"
            )
            self.state = VADState.SPEECH
            self.silence_length = 0.0
            self.speech_length += self.frame_length
            # Forward to next modules
            output_iu = self.create_iu()
            output_iu.set_audio(iu.raw_audio, iu.nframes, iu.rate, iu.sample_width)
            return retico_core.UpdateMessage.from_iu(
                output_iu, retico_core.UpdateType.ADD
            )
        else:  # Silence
            self.silence_length += self.frame_length
            # case 01: Silence inside A TURN
            if self.state == VADState.SILENCE_TURN:
                # case 01.1: Reached max silence allowed inside a single turn => End of Turn detected
                if self.silence_length > self.max_silence_length:
                    if self.speech_length >= self.min_turn_length:
                        logging.info(
                            f"{ConsoleColors.BLUE}VAD:{self.state}:{ConsoleColors.RESET}, Reached Max Silence Length, End of Turn Detected , Turn length = {self.speech_length}"
                        )
                        self.silence_length = 0.0
                        self.speech_length = 0.0
                        self.state = VADState.SILENCE
                        output_iu = self.create_iu()
                        output_iu.set_audio(
                            iu.raw_audio, iu.nframes, iu.rate, iu.sample_width
                        )
                        return retico_core.UpdateMessage.from_iu(
                            output_iu, retico_core.UpdateType.COMMIT
                        )
                    else:
                        logging.debug(
                            f"{ConsoleColors.MAGENTA}VAD:{self.state}:{ConsoleColors.RESET}, Silence Detected, Reached Max Silence Length, End of Turn Detected but turn too short"
                        )
                        # Not enough speech to commit => Revoke the turn (noise...)
                        self.silence_length = 0.0
                        self.speech_length = 0.0
                        self.state = VADState.SILENCE
                        output_iu = self.create_iu()
                        output_iu.set_audio(
                            iu.raw_audio, iu.nframes, iu.rate, iu.sample_width
                        )
                        return retico_core.UpdateMessage.from_iu(
                            output_iu, retico_core.UpdateType.REVOKE
                        )
                else:
                    logging.debug(
                        f"{ConsoleColors.MAGENTA}VAD:{self.state}:{ConsoleColors.RESET}, Silence Detected, Keep Buffering Silence silence_length = {self.silence_length}, speech_length = {self.speech_length}"
                    )
                    self.state = VADState.SILENCE_TURN
                    output_iu = self.create_iu()
                    output_iu.set_audio(
                        iu.raw_audio, iu.nframes, iu.rate, iu.sample_width
                    )
                    return retico_core.UpdateMessage.from_iu(
                        output_iu, retico_core.UpdateType.ADD
                    )
            # case 02: started detecting silence inside a turn
            elif self.state == VADState.SPEECH:
                logging.debug(
                    f"{ConsoleColors.MAGENTA}VAD:{self.state}:{ConsoleColors.RESET}, Silence Detected, Started Silence State"
                )
                self.silence_length += self.frame_length
                self.state = VADState.SILENCE_TURN
                output_iu = self.create_iu()
                output_iu.set_audio(iu.raw_audio, iu.nframes, iu.rate, iu.sample_width)
                return retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )
            elif self.state == VADState.SILENCE:
                return None  # Not Inside a turn, ignore the audio
