# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import subprocess
from cog import BasePredictor, Input
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

AYA_CACHE_DIR = "cache"
AYA_MODEL_NAME = "models--CohereForAI--aya-101"
AYA_URL = f"https://weights.replicate.delivery/default/aya-101/{AYA_MODEL_NAME}.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-vx", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model and tokenizer into memory to make running multiple predictions efficient"""
        model_weights_path = os.path.join(AYA_CACHE_DIR, AYA_MODEL_NAME)
        if not os.path.exists(model_weights_path):
            print(f"Downloading model weights to {model_weights_path}...")
            download_weights(AYA_URL, model_weights_path)

        checkpoint = "CohereForAI/aya-101"
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            cache_dir=AYA_CACHE_DIR,
            local_files_only=True,
        )
        self.aya_model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            cache_dir=AYA_CACHE_DIR,
            local_files_only=True,
        )

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for completion",
            default="Translate to English: Aya cok dilli bir dil modelidir.",
        ),
    ) -> str:
        """Generate a completion for the given text prompt"""
        encoded_prompt = self.tokenizer.encode(prompt, return_tensors="pt")
        generated_tokens = self.aya_model.generate(encoded_prompt, max_new_tokens=128)
        completion = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )
        return completion
