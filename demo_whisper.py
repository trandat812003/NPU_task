import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_trained = WhisperForConditionalGeneration.from_pretrained(
    "hkab/whisper-base-vietnamese-finetuned"
)
processor = WhisperProcessor.from_pretrained("hkab/whisper-base-vietnamese-finetuned")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")

input_speech, rate = librosa.load(
    "/home/trandat/Documents/vnpt/NPU_task/demo.wav", sr=16000
)
input_features = processor(
    input_speech, sampling_rate=rate, return_tensors="pt"
).input_features

predicted_ids = model_trained.generate(
    input_features, forced_decoder_ids=forced_decoder_ids
)

print(f"Prediction: {processor.batch_decode(predicted_ids, skip_special_tokens=True)}")
