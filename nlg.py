import time
import numpy as np
import retico_core
from retico_core.audio import AudioIU
from retico_core.text import TextIU
import logging
from console_colors import ConsoleColors
from openai import OpenAI
from transformers import AutoTokenizer

from abc import ABC, abstractmethod


class NLG(retico_core.AbstractModule, ABC):
    """NLG Module, manages dialogue history and generates responses."""

    @staticmethod
    def name():
        return "Natural Language Generation Module"

    @staticmethod
    def description():
        return "A module that generates responses based on the dialogue history and updates the dialogue state."

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dialogue_history = []

    def setup(self):
        pass

    def process_update(self, update_message):
        iu, ut = next(update_message)
        if ut == retico_core.UpdateType.COMMIT:
            start_time = time.time()

            # Update dialogue history with the user input
            self.dialogue_history.append({"role": "user", "content": iu.text})

            # Generate the output using the dialogue history
            output_text = self.generate_response()
            # Send to the next modules
            output_iu = self.create_iu()
            output_iu.set_text(output_text)
            out_message = retico_core.UpdateMessage.from_iu(
                output_iu, retico_core.UpdateType.COMMIT
            )
            self.append(out_message)
            self.dialogue_history.append({"role": "assistant", "content": output_text})
            end_time = time.time()
            logging.info(
                f"{ConsoleColors.BLUE}NLG:{ConsoleColors.RESET} NLG Inference time : {end_time-start_time}, NLG Output : {output_text}"
            )

    @abstractmethod
    def generate_response(self, user_input):
        pass


class OpenAINLG(NLG):
    """NLG Module using OpenAI API for response generation.

    VLLM can be started with : vllm serve meta-llama/Llama-3.2-1B-Instruct --dtype auto --api-key token-abc123 --dtype=half --max-model-len 16384

    """

    def __init__(self, api_key, api_base, model_id, inference_args, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_base = api_base
        self.model_id = model_id
        self.inference_args = inference_args

    def setup(self):
        self.dialogue_history.append(
            {
                "role": "system",
                "content": "Pretend to be a human discussing with his friend. Always answer using short sentences abd be proactive during the convarsation.",
            }
        )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        models = [model.id for model in self.client.models.list()]
        logging.debug(
            f"{ConsoleColors.MAGENTA}OpenAINLG:{ConsoleColors.RESET} Available models : {models}"
        )
        if self.model_id not in models:
            logging.error(
                f"{ConsoleColors.RED}OpenAINLG:{ConsoleColors.RESET} : {self.model_id} not running !! "
            )
            raise RuntimeError("OpenAINLG Error")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logging.info(
            f"{ConsoleColors.BLUE}OpenAINLG:{ConsoleColors.RESET} Module setup done"
        )

    def generate_response(self):
        prompt = self.tokenizer.apply_chat_template(
            self.dialogue_history,
            tokenize=False,  # To get Raw String
            add_generation_prompt=True,  # Adds an empty assistant turn at the end
        )
        logging.debug(
            f"{ConsoleColors.MAGENTA}OpenAINLG:{ConsoleColors.RESET} : Calling API with prompt : {prompt!r}"
        )
        completion = self.client.completions.create(
            model=self.model_id,
            prompt=prompt,
            echo=False,
            stream=False,
            **self.inference_args,
        )
        return completion.choices[0].text
