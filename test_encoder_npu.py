import torch, time
import onnxruntime as ort
import torchaudio
from transformers import WhisperForConditionalGeneration, AutoTokenizer, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput


onnx_path = "model/encoder/model.onnx"
session = ort.InferenceSession(onnx_path, providers=["QNNExecutionProvider"])

model_name = "hkab/whisper-base-vietnamese-finetuned"
processor = WhisperProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
decoder = WhisperForConditionalGeneration.from_pretrained(model_name)

wav_path = r"C:\Users\asus\Documents\datnt\VIVOSDEV01\VIVOSDEV01_R002.wav"
waveform, sr = torchaudio.load(wav_path)


waveform = waveform.mean(dim=0).numpy()
inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
input_features = inputs.input_features.numpy().astype("float32")  

# torch.save(inputs, "inputs.npy")
start_time = time.time()
for i in range(30):
    # start_time = time.time()
    input_name = session.get_inputs()[0].name
    encoder_outputs = session.run(None, {input_name: input_features})

    encoder_hidden_states = torch.from_numpy(encoder_outputs[0]).to(torch.float32)

    encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
    generated_ids = decoder.generate(
        encoder_outputs=encoder_outputs,
        max_length=200
    )

    decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # end_time = time.time()
    print("Decoded text:", decoded_text)
    print("done")

print(f"Time: {time.time() - start_time:.3f} seconds")
