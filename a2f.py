import retico_core
from retico_core.audio import AudioIU
from retico_core.text import TextIU
from console_colors import ConsoleColors


from audio2face_api.A2F import Audio2FaceStream

import logging


class A2FStream(retico_core.AbstractConsumingModule):
    """A module to stream the agents speech for audio play and non-verbal generation to Audio2Face
    Use gRPC to stream the audio
    """

    @staticmethod
    def name():
        return "Audio2Face Streamer"

    @staticmethod
    def description():
        return "A module to stream the agents speech for audio play and non-verbal gheneration"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return None

    def __init__(
        self,
        scene_path="./assets/mark_solved_streaming.usd",
        api_url="http://localhost:8011",
        grpc_url="localhost:50051",
        fps=30,
        chunck_size=4000,
        use_keyframes=True,
        use_global_emotion=True,
        global_emotion={"joy": 0.9, "sadness": 0.1},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.a2f = Audio2FaceStream(
            grpc_url=grpc_url,
            chunk_size=chunck_size,
            block_until_playback_is_finished=False,
            use_livelink=True,  # Set to true to receive the generated frames from Aufio2Face
            api_url=api_url,
            scene_path=scene_path,
            fps=fps,
            use_keyframes=use_keyframes,
            use_global_emotion=use_global_emotion,
            global_emotion=global_emotion,
        )
        self.use_keyframes = use_keyframes
        self.use_global_emotion = use_global_emotion
        self.global_emotion = global_emotion

    def setup(self):
        self.a2f.init_A2F()
        self.a2f.a2e.set_auto_emotion_detect(auto_detect=self.use_keyframes)
        if self.use_global_emotion:
            self.a2f.a2e.set_gloabl_emotion(joy=0.95, amazement=0.05)
        logging.info(
            f"{ConsoleColors.BLUE}A2FStream:{ConsoleColors.RESET} Setup Completed"
        )

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD or ut == retico_core.UpdateType.COMMIT:
                frames = self.a2f.stream_audio(iu.raw_audio, iu.rate)
                logging.debug(
                    f"{ConsoleColors.BLUE}A2FStream:{ConsoleColors.RESET} Streamed an audio"
                )
        return None

    def shutdown(self):
        logging.info(
            f"{ConsoleColors.BLUE}A2FStream:{ConsoleColors.RESET} Shutting Down"
        )
        self.a2f.end_a2f_connection()
