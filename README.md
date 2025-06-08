# S2S_Conversational_Agent

An **incremental Speech-to-Speech Conversational Agent** built using the [Retico](https://github.com/retico-team/retico) framework. This system supports streaming conversations with real-time VAD, ASR, NLG, and TTS. Optionally, it integrates with NVIDIA Audio2Face for facial animation from speech

## Modules

- **VAD Module** : `webrtcvad` – detects speech segments.
- **ASR Module** : `openai/whisper-large-v3-turbo` – converts speech to text.
- **NLG Module** : Uses OpenAI-compatible LLM API (e.g. via `vllm`) for natural language generation.
- **TTS Module** : Uses [Kokoro-82M]() for real-time speech synthesis.
- **A2F Integration** : Sends audio to NVIDIA Audio2Face for real-time 3D avatar animation, using the [audio2face_api implementation](https://github.com/oussama-sil/audio2face_api).

## Installation

Create and activate your Python environment, then install dependencies:

```
pip install -r requirements.txt

```

## Quickstart

### 1. Start a vLLM server

```
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --api-key token-abc123 \
  --dtype=half \
  --max-model-len 4096
```

### 2. Run the application

```
python main.py

```

### Optional: With Audio2Face

```
python main.py --use_a2f

```

You can also pass other arguments:

```
python main.py \
  --nlg_api_key token-abc123 \
  --nlg_api_base http://localhost:8000/v1 \
  --nlg_model_id meta-llama/Llama-3.2-1B-Instruct \
  --a2f_api_url http://localhost:8011 \
  --a2f_grpc_url localhost:50051 \
  --use_a2f

```
