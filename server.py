import argparse
import io
import json
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from kserve import InferRequest, InferResponse, Model, ModelServer, model_server
from kserve.errors import InvalidInput

class WhisperModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.model_id = args.model_id or "openai/whisper-tiny.en"
        self.device = torch.device(args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.ready = False
        self.load()

    def load(self):
        # Load the Whisper model and processor
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.ready = True

    def preprocess(self, payload: bytes, headers: Dict[str, str] = None) -> Dict:
        # Decode the bytes payload into JSON format
        try:
            payload_str = payload.decode('utf-8')
            payload_dict = json.loads(payload_str)
        except Exception as e:
            raise InvalidInput(f"Invalid payload format: {e}")

        # Ensure that the payload contains an audio file
        if "instances" in payload_dict and "audio" in payload_dict["instances"][0]:
            audio_bytes = payload_dict["instances"][0]["audio"]
        else:
            raise InvalidInput("Audio file is missing in the payload.")

        # Convert audio bytes to waveform
        audio_io = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_io)

        return {"waveform": waveform, "sample_rate": sample_rate}

    def predict(self, payload: bytes, headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        # Preprocess the audio file
        processed_input = self.preprocess(payload, headers)
        waveform = processed_input["waveform"]
        sample_rate = processed_input["sample_rate"]

        # Process input features for the Whisper model
        input_features = self.processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features

        # Generate transcription
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return {
            "predictions": [
                {
                    "transcription": transcription
                }
            ]
        }

# Argument parser for model options
parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_id",
    type=str,
    help="Model ID to load (default: openai/whisper-tiny.en)"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to use for inference, valid values: 'cuda' (default), 'cpu'"
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    # Initialize and load the Whisper model
    model = WhisperModel("whisper_model")
    model.load()

    # Start KServe ModelServer
    ModelServer().start([model])
