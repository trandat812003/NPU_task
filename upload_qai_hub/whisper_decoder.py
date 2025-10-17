import torch
from whisper.model import Whisper, ModelDimensions
import qai_hub as hub

ckpt_path = r"C:\Users\asus\Documents\datnt\NPU_task\whisper.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

dims = ModelDimensions(**checkpoint["dims"])
model = Whisper(dims)
model.load_state_dict(checkpoint["model_state_dict"])
decoder = model.decoder
decoder.eval()

x = torch.randint(0, 51864, (1, 1), dtype=torch.long)
xa = torch.randn((1, 1500, 512), dtype=torch.float32)


torch.onnx.export(
    decoder,
    (x, xa),
    "whisper_decoder.onnx",
    input_names=["x", "xa"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "x": {0: "batch", 1: "tokens"},
        "xa": {0: "batch", 1: "frames"},
    }
)
print("✅ Exported whisper_decoder.onnx")

compile_job = hub.submit_compile_job(
    model="whisper_decoder.onnx",
    device=hub.Device("Snapdragon X Elite CRD"),
    input_specs={
        "x": ((1, 1), "int64"),
        "xa": ((1, 1500, 512), "float32"),
    },
    options="--target_runtime qnn_context_binary --truncate_64bit_io"
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
target_model.download("whisper-decoder")
print("🎯 Done! whisper-decoder downloaded.")
