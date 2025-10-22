import torch
import qai_hub as hub
from qai_hub_models.models._shared.whisper.model import (
    CollectionModel,
    Whisper,
    WhisperDecoderInf,
    WhisperEncoderInf,
)
from qai_hub_models.utils.base_model import TargetRuntime, Precision


# ===============================
# 1️⃣ Định nghĩa class WhisperBaseEn
# ===============================
@CollectionModel.add_component(WhisperEncoderInf)
@CollectionModel.add_component(WhisperDecoderInf)
class WhisperBaseEn(Whisper):
    @classmethod
    def from_pretrained(cls):
        return super().from_pretrained(
            r"C:\Users\asus\Documents\datnt\NPU_task\whisper.pt"
        )


# ===============================
# 2️⃣ Load mô hình
# ===============================
whisper_base_en = WhisperBaseEn.from_pretrained()

encoder = whisper_base_en.encoder
decoder = whisper_base_en.decoder

print("✅ Loaded Whisper model successfully.")


# ===============================
# 3️⃣ Tạo dummy input để trace
# ===============================
dummy_audio = torch.randn(1, 80, 3000)  # mel features
dummy_input_ids = torch.tensor([[50258]])  # SOT token
dummy_index = torch.tensor([[0]])
dummy_k_cache_cross = torch.zeros((6, 8, 64, 1500))
dummy_v_cache_cross = torch.zeros((6, 8, 1500, 64))
dummy_k_cache_self = torch.zeros((6, 8, 64, 224))
dummy_v_cache_self = torch.zeros((6, 8, 224, 64))

print("🔧 Tracing models...")


# ===============================
# 4️⃣ Trace encoder và decoder
# ===============================
traced_encoder = torch.jit.trace(encoder, dummy_audio)
traced_decoder = torch.jit.trace(
    decoder,
    (
        dummy_input_ids,
        dummy_index,
        dummy_k_cache_cross,
        dummy_v_cache_cross,
        dummy_k_cache_self,
        dummy_v_cache_self,
    ),
)

# Lưu file traced
traced_encoder.save("whisper_encoder_traced.pt")
traced_decoder.save("whisper_decoder_traced.pt")

encoder_options = whisper_base_en.encoder.get_hub_compile_options(
    target_runtime=TargetRuntime.QNN_CONTEXT_BINARY,
    precision=Precision.float,
)

decoder_options = whisper_base_en.decoder.get_hub_compile_options(
    target_runtime=TargetRuntime.QNN_CONTEXT_BINARY,
    precision=Precision.float,
)


encoder_job = hub.submit_compile_job(
    model="whisper_encoder_traced.pt",
    options=encoder_options,
    input_specs={
        "audio": ((1, 80, 3000), "float32"),  # tensor input của encoder
    },
    # output_names=["k_cache", "v_cache"],
    device=hub.Device("Snapdragon X Elite CRD"),
    name="whisper_encoder_compile",
)

decoder_job = hub.submit_compile_job(
    model="whisper_decoder_traced.pt",
    options=decoder_options,
    input_specs={
        "x": ((1, 1), "int32"), 
        "index": ((1, 1), "int32"),
        "k_cache_cross": ((6, 8, 64, 1500), "float32"),
        "v_cache_cross": ((6, 8, 1500, 64), "float32"),
        "k_cache_self": ((6, 8, 64, 224), "float32"),
        "v_cache_self": ((6, 8, 224, 64), "float32"),
    },
    device=hub.Device("Snapdragon X Elite CRD"),
    name="whisper_decoder_compile",
)

target_encoder = encoder_job.get_target_model()
encoder_output_path = target_encoder.download("whisper_encoder")
target_decoder = decoder_job.get_target_model()
decoder_output_path = target_decoder.download("whisper_decoder")

print("✅ Done! Models compiled successfully:")
print("Encoder saved to:", encoder_output_path)
print("Decoder saved to:", decoder_output_path)

