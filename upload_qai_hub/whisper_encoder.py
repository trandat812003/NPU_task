import torch

import qai_hub as hub

input_shape = (1, 80, 3000)
example_input = torch.rand(input_shape)


onnx_path = r"C:\Users\asus\Documents\datnt\NPU_task\whisper_onnx\encoder_model.onnx"
compile_job = hub.submit_compile_job(
    model=onnx_path,
    device=hub.Device("Snapdragon X Elite CRD"),
    input_specs=dict(image=input_shape),
    options="--target_runtime qnn_context_binary"
)
assert isinstance(compile_job, hub.CompileJob)

# Step 3: Profile on cloud-hosted device
target_model = compile_job.get_target_model()
assert isinstance(target_model, hub.Model)
profile_job = hub.submit_profile_job(
    model=target_model,
    device=hub.Device("Snapdragon X Elite CRD"),
)
assert isinstance(profile_job, hub.ProfileJob)
print(profile_job)

target_model = compile_job.get_target_model()
assert isinstance(target_model, hub.Model)
target_model.download("whisper-encoder")