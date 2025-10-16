import torch
import qai_hub as hub

batch_size = 1
decoder_seq_len = 80
encoder_seq_len = 3000

onnx_path = r"C:\Users\asus\Documents\datnt\NPU_task\whisper_onnx\decoder_model.onnx"

compile_job = hub.submit_compile_job(
    model=onnx_path,
    device=hub.Device("Snapdragon X Elite CRD"),
    input_specs={
        "input_ids": ((batch_size, decoder_seq_len), "int64"),
        "encoder_hidden_states": ((batch_size, encoder_seq_len // 2, 512), "float32"),
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


target_model.download("whisper-decoder")
