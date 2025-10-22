import os
import time
import json
import torch
import numpy as np
import samplerate
import audio2numpy as a2n
import argparse
import pandas as pd
from scipy import special as scipy_special  # type: ignore
from transformers import WhisperProcessor, WhisperTokenizer
from datasets import load_metric
from pathlib import Path

from qai_hub_models.models._shared.hf_whisper.model import (
    CHUNK_LENGTH,
    SAMPLE_RATE,
    MASK_NEG,
    MEAN_DECODE_LEN,
)
from qai_appbuilder import QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig


# ============================
# 1️⃣ LỚP ENCODER / DECODER
# ============================

class Encoder(QNNContext):
    def Inference(self, input_data):
        input_datas = [input_data]
        output_data = super().Inference(input_datas)
        num_layers = len(output_data) // 2  # = 12

        kv_cache_cross = []
        for i in range(num_layers):
            k = output_data[2 * i]
            v = output_data[2 * i + 1]
            k = k.reshape(8, 1, 64, 1500)
            v = v.reshape(8, 1, 1500, 64)
            kv_cache_cross.append((k, v))
        return tuple(kv_cache_cross)


class Decoder(QNNContext):
    def Inference(self, *input_datas):
        output_data = super().Inference(input_datas)
        return output_data


# ============================
# 2️⃣ CẤU HÌNH QNN
# ============================

QNNConfig.Config(
    r"C:\Users\asus\Documents\datnt\NPU_task\demo_qualcomm\qai_libs",
    Runtime.HTP,
    LogLevel.WARN,
    ProfilingLevel.BASIC
)

decoder = Decoder(
    "whisper_decoder",
    r"C:\Users\asus\Documents\datnt\NPU_task\demo_qualcomm\models\HfWhisperDecoder.bin"
)
encoder = Encoder(
    "whisper_encoder",
    r"C:\Users\asus\Documents\datnt\NPU_task\demo_qualcomm\models\HfWhisperEncoder.bin"
)
processor = WhisperProcessor.from_pretrained("hkab/whisper-base-vietnamese-finetuned")
tokenizer = WhisperTokenizer.from_pretrained("hkab/whisper-base-vietnamese-finetuned")

CONFIG_PATH = r"C:\Users\asus\Documents\datnt\NPU_task\config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config_dict = json.load(f)


# ============================
# 3️⃣ HÀM TIỀN XỬ LÝ ÂM THANH
# ============================

def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate=SAMPLE_RATE,
    model_chunk_seconds=CHUNK_LENGTH,
) -> list[np.ndarray]:
    if audio_sample_rate != model_sample_rate:
        audio = samplerate.resample(audio, model_sample_rate / audio_sample_rate)
        audio_sample_rate = model_sample_rate

    num_chunks = audio.shape[0] // (audio_sample_rate * model_chunk_seconds)
    if num_chunks == 0:
        return [audio]

    last_sample = audio_sample_rate * num_chunks * model_chunk_seconds
    return [
        *np.array_split(audio[:last_sample], num_chunks),
        audio[last_sample:],
    ]


# ============================
# 4️⃣ HÀM DECODE 1 CHUNK
# ============================

def transcribe_single_chunk(audio: np.ndarray):
    input_features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.cpu().numpy()

    kv_cache_cross = encoder.Inference(input_features)

    sot = config_dict.get("decoder_start_token_id", None)
    num_decoder_blocks = config_dict.get("decoder_layers", None)
    attention_dim = config_dict.get("d_model", None)
    num_decoder_heads = config_dict.get("decoder_attention_heads", None)
    mask_neg = config_dict.get("mask_neg", MASK_NEG)
    eot = config_dict.get("eos_token_id", None)

    output_ids = torch.tensor([[sot]])
    output_logits = []
    output_length = output_ids.shape[1]

    position_ids = torch.tensor([0], dtype=torch.int32)
    attention_mask = torch.full((1, 1, 1, MEAN_DECODE_LEN), mask_neg, dtype=torch.float32)

    k_cache_self = torch.zeros((num_decoder_heads, 1, attention_dim // num_decoder_heads, MEAN_DECODE_LEN - 1), dtype=torch.float32)
    v_cache_self = torch.zeros((num_decoder_heads, 1, MEAN_DECODE_LEN - 1, attention_dim // num_decoder_heads), dtype=torch.float32)
    kv_cache_self = tuple((k_cache_self, v_cache_self) for _ in range(num_decoder_blocks))

    for n in range(MEAN_DECODE_LEN - 1):
        input_ids = output_ids[:, n:n+1].to(torch.int32)
        attention_mask[:, :, :, MEAN_DECODE_LEN - n - 1] = 0.0

        flattened_kv_cache_self = tuple(item for sublist in kv_cache_self for item in sublist)
        flattened_kv_cache_cross = tuple(item for sublist in kv_cache_cross for item in sublist)

        decoder_input = (input_ids, attention_mask) + flattened_kv_cache_self + flattened_kv_cache_cross + (position_ids,)
        decoder_output = decoder.Inference(*decoder_input)

        if isinstance(decoder_output, tuple) and len(decoder_output) == 2:
            logits, kv_cache_self = decoder_output
        else:
            logits = decoder_output[0]
            kv_cache_self = tuple(decoder_output[i:i+2] for i in range(1, len(decoder_output), 2))

        logits = logits.reshape(1, 51865, 1, 1)
        logits = torch.from_numpy(logits)

        output_logits.append(logits.detach().clone())
        output_id = torch.argmax(logits, 1).squeeze(0)

        if output_id == eot:
            output_ids = torch.cat((output_ids, output_id), -1)
            break
        if n >= output_length - 1:
            output_ids = torch.cat((output_ids, output_id), -1)

        position_ids += 1

    return output_ids[0].tolist()


# ============================
# 5️⃣ HÀM INFERENCE TRẢ TEXT
# ============================

def run_inference_return_text(audio_path: str) -> str:
    audio, sr = a2n.audio_from_file(audio_path)
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

    chunks = chunk_and_resample_audio(audio, sr)
    tokens = []
    for ch in chunks:
        tokens.extend(transcribe_single_chunk(ch))

    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    PerfProfile.RelPerfProfileGlobal()
    return text


# ============================
# 6️⃣ ĐÁNH GIÁ CẢ THƯ MỤC
# ============================

def evaluate_folder(audio_folder: str, prompt_path: str):
    # Đọc file prompt.txt
    with open(prompt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    gt_dict = {}
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            fname, text = parts
            gt_dict[f"{fname.strip()}.wav"] = text.strip()

    wer_metric = load_metric("wer")
    preds, refs, names, times = [], [], [], []

    for fname in os.listdir(audio_folder):
        if not fname.lower().endswith(".wav"):
            continue
        audio_path = os.path.join(audio_folder, fname)

        start_time = time.perf_counter()
        pred = run_inference_return_text(audio_path)
        elapsed = time.perf_counter() - start_time

        ref = gt_dict.get(fname, "")

        preds.append(pred.lower().strip())
        refs.append(ref.lower().strip())
        names.append(fname.lower().strip())
        times.append(elapsed)

        print(f"\n🎧 File: {fname}")
        print(f"  🗣 Predicted: {pred}")
        print(f"  💬 Ground Truth: {ref}")
        print(f"  ⏱️ Time: {elapsed:.2f} s")

    # Tính WER
    wer = wer_metric.compute(predictions=preds, references=refs)

    # Thống kê thời gian
    min_t = np.min(times)
    max_t = np.max(times)
    avg_t = np.mean(times)

    print("\n==========================")
    print(f"Word Error Rate (WER): {wer:.2f}")
    print(f"Min Inference Time: {min_t:.2f} s")
    print(f"Max Inference Time: {max_t:.2f} s")
    print(f"Avg Inference Time: {avg_t:.2f} s")
    print("==========================")

    df = pd.DataFrame({
        "filename": names,
        "predicted": preds,
        "ground_truth": refs,
        "inference_time_s": times
    })
    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    print("📁 Kết quả đã lưu tại: results.csv")


# ============================
# 7️⃣ MAIN
# ============================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper NPU on a folder")
    parser.add_argument(
        "--audio_folder",
        type=str,
        default=r"C:\Users\asus\Documents\VIVOSDEV01",
        help="Thư mục chứa file .wav"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=r"C:\Users\asus\Documents\prompts.txt",
        help="File chứa ground truth dạng: <filename> <text>"
    )
    args = parser.parse_args()
    evaluate_folder(args.audio_folder, args.prompt_path)


if __name__ == "__main__":
    main()
