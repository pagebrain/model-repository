# BASED ON https://huggingface.co/spaces/coqui/xtts/blob/main/app.py
from cog import BasePredictor, Input, Path
import torch
import os
# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


from scipy.io import wavfile
class Predictor(BasePredictor):

    def setup(self):
        self.config = XttsConfig()
        self.config.load_json("/model/config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir="/model/", eval=True)
        self.model.cuda()

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Text prompt",
            default=
            "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        ),
        language: str = Input(
            description="Output language for the synthesised speech",
            choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"],
            default="en",
        ),
        speaker_wav: Path = Input(description="Reference audio")
    ) -> Path:
        if len(prompt) < 2:
            raise ValueError("Please give a longer prompt text")
        #speaker_wav = "/male.wav"
        output = self.model.synthesize(prompt,
            self.config,
            speaker_wav=speaker_wav,
            language=language,
        )
        wavfile.write("/tmp/output.wav", self.config.audio.sample_rate, output['wav'])
        return Path("/tmp/output.wav")
