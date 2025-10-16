import torch
import qai_hub as hub

batch_size = 1
decoder_seq_len = 10
encoder_seq_len = 3000

input_ids = torch.randint(0, 50257, (batch_size, decoder_seq_len), dtype=torch.long)
encoder_hidden_states = torch.randn(batch_size, encoder_seq_len // 2, 512, dtype=torch.float32)

onnx_path = r"C:\Users\asus\Documents\datnt\NPU_task\whisper_onnx\decoder_model.onnx"

compile_job = hub.submit_compile_job(
    model=onnx_path,
    device=hub.Device("Snapdragon X Elite CRD"),
    input_specs=dict(
        input_ids=input_ids.shape,
        encoder_hidden_states=encoder_hidden_states.shape
    ),
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


target_model.download("whisper-decoder")
