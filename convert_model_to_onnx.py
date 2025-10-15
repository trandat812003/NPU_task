from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

model = ORTModelForSpeechSeq2Seq.from_pretrained("hkab/whisper-base-vietnamese-finetuned", export=True)
model.save_pretrained("./whisper_onnx")
