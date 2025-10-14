from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

model = ORTModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small", export=True)
model.save_pretrained("./whisper_onnx")
