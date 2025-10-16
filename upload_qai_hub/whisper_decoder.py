import torch

import qai_hub as hub

batch_size = 1
seq_len = 1       
time_steps = 1500  
hidden_size = 512  

dummy_decoder_input_ids = torch.randint(
    low=0, high=50257, size=(batch_size, seq_len), dtype=torch.long
)
dummy_encoder_hidden_states = torch.rand(
    batch_size, time_steps, hidden_size, dtype=torch.float32
)
onnx_path = r"C:\Users\asus\Documents\datnt\whisper-base-vietnamese-onnx\decoder_model.onnx"

compile_job = hub.submit_compile_job(
    model=onnx_path,
    device=hub.Device("Snapdragon X Elite CRD"),
    input_specs={
        "input_ids": (dummy_decoder_input_ids.shape, "int64"),
        "encoder_hidden_states": (dummy_encoder_hidden_states.shape, "float32")
    },
    options="--target_runtime onnx"
)
assert isinstance(compile_job, hub.CompileJob)
target_model = compile_job.get_target_model()
assert isinstance(target_model, hub.Model)

profile_job = hub.submit_profile_job(
    model=target_model,
    device=hub.Device("Snapdragon X Elite CRD"),
)
assert isinstance(profile_job, hub.ProfileJob)
print(profile_job)

target_model.download("whisper-decoder-onnx")