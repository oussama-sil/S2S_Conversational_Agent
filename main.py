import time
import retico_core
from retico_core.audio import MicrophoneModule, SpeakerModule
import torch
from vad import VAD
from asr import ASR
from nlg import OpenAINLG
from tts import TTS
from a2f import A2FStream
import argparse


import logging

logging.basicConfig(level=logging.INFO)

# If set to True, TTS will stream its output to the A2F instead of the speaker


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":

    logging.info("[MAIN] Starting Retico System")

    parser = argparse.ArgumentParser()

    parser.add_argument("--nlg_api_key", type=str, default="token-abc123")
    parser.add_argument("--nlg_api_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument(
        "--nlg_model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument("--a2f_api_url", type=str, default="http://localhost:8011")
    parser.add_argument("--a2f_grpc_url", type=str, default="localhost:50051")
    parser.add_argument("--use_a2f", action="store_true", help="Enable A2F usage")

    args = parser.parse_args()

    # ? Input
    frame_length = 0.02  # Length of one Audio Frame
    sample_rate = 16000
    microphone_module = MicrophoneModule(
        frame_length=frame_length,
        rate=sample_rate,
    )

    # ? VAD
    vad_module = VAD(
        mode=3,
        sample_rate=sample_rate,
        max_silence_length=0.700,
        frame_length=frame_length,
        min_turn_length=0.150,
    )
    # ? ASR
    asr_module = ASR(
        model_id="openai/whisper-large-v3-turbo",
        device=device,
    )

    inference_args = {"max_tokens": 250, "stop": ["<|eot_id|>"]}
    nlg_module = OpenAINLG(
        api_key=args.nlg_api_key,
        api_base=args.nlg_api_base,
        model_id=args.nlg_model_id,
        inference_args=inference_args,
    )

    # A2F
    try:
        # Audio2Face Streamer
        a2f_module = A2FStream(
            scene_path="./assets/mark_solved_streaming.usd",
            api_url=args.a2f_api_url,
            grpc_url=args.a2f_grpc_url,
            fps=30,
            chunck_size=4000,
            use_keyframes=True,
            use_global_emotion=False,
            global_emotion={"joy": 0.9, "sadness": 0.1},
        )
        a2f_module.setup()
        logging.info("[MAIN] A2F initialized")
    except Exception:
        logging.error(
            "[MAIN] Unable to initialize Audio2Face, falling back to SpeakerModule"
        )
        args.use_a2f = False

    # TTS
    tts_sample_rate = 24000
    tts_model_args = {
        "device": device,
        "lang_code": "a",
        "repo_id": "hexgrad/Kokoro-82M",
    }
    tts_module = TTS(
        sample_rate=tts_sample_rate,
        model_args=tts_model_args,
        output_audio_bytes=not args.use_a2f,
    )

    # Speaker
    speaker_module = SpeakerModule(rate=tts_sample_rate)

    microphone_module.subscribe(vad_module)
    vad_module.subscribe(asr_module)
    asr_module.subscribe(nlg_module)
    nlg_module.subscribe(tts_module)

    if args.use_a2f:
        logging.info("[MAIN] Using Audio2Face")
        tts_module.subscribe(a2f_module)
    else:
        logging.info("[MAIN] Using Speaker Module")

        tts_module.subscribe(speaker_module)

    retico_core.network.run(microphone_module)
try:
    input()
    microphone_module.stop()
    vad_module.stop()
    nlg_module.stop()
    tts_module.stop()
    asr_module.stop()
    speaker_module.stop()
    a2f_module.stop()
except (KeyboardInterrupt, AttributeError) as e:
    microphone_module.stop()
    vad_module.stop()
    nlg_module.stop()
    tts_module.stop()
    asr_module.stop()
    speaker_module.stop()
    a2f_module.stop()
