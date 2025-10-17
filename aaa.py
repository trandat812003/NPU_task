# test_whisper_local.py
import whisper
import torch
import sys

model_path = r"C:\Users\asus\Documents\datnt\NPU_task\whisper.pt"  
audio_path = r"C:\Users\asus\Documents\datnt\NPU_task\demo.wav"  

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

try:
    # try load model from local .pt
    model = whisper.load_model(model_path, device=device)
except Exception as e:
    print("Error when loading with whisper.load_model():", e)
    print("Falling back to torch.load inspection...")
    import torch
    sd = torch.load(model_path, map_location="cpu")
    print("Top-level keys in the .pt file:", list(sd.keys())[:50])
    sys.exit(1)

print("Model loaded. Transcribing", audio_path)
result = model.transcribe(audio_path, language="vi", task="transcribe")
print("=== Transcript ===")
print(result["text"])
