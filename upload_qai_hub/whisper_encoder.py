import torch
from whisper.model import Whisper, ModelDimensions
import qai_hub as hub

input_shape = (1, 80, 3000)

ckpt_path = r"C:\Users\asus\Documents\datnt\NPU_task\whisper.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

dims = ModelDimensions(**checkpoint["dims"])
model = Whisper(dims)
model.load_state_dict(checkpoint["model_state_dict"])

encoder = model.encoder
encoder.eval()


dummy_input = torch.randn(1, 80, 3000)

torch.onnx.export(
    encoder,
    (dummy_input),
    "whisper_encoder.onnx",
    input_names=["input_features"],
    output_names=["encoder_hidden_states"],
    opset_version=17,
)
print("✅ Exported whisper_encoder.onnx")

print("Tracing Whisper encoder ...")
traced_encoder = torch.jit.trace(encoder, dummy_input)

compile_job = hub.submit_compile_job(
    model="whisper_encoder.onnx",
    device=hub.Device("Snapdragon X Elite CRD"),
    input_specs=dict(input_features=input_shape),
    options="--target_runtime qnn_context_binary"
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

target_model = compile_job.get_target_model()
assert isinstance(target_model, hub.Model)
target_model.download("whisper-encoder")