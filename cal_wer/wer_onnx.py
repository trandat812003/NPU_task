import os
import glob
import numpy as np
import librosa
import onnxruntime as ort
from transformers import WhisperProcessor
import evaluate
import pandas as pd

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# ==========================
# 1. Hàm xử lý âm thanh & mô hình
# ==========================
def extract_features(wav_path: str, processor: WhisperProcessor):
    audio, sr = librosa.load(wav_path, sr=16000)
    feats = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="np")
    return feats["input_features"].astype(np.float32)

def run_encoder(encoder_sess: ort.InferenceSession, input_features: np.ndarray):
    name = encoder_sess.get_inputs()[0].name
    outs = encoder_sess.run(None, {name: input_features})
    return outs[0]

def run_decoder_once(decoder_sess: ort.InferenceSession, input_ids: np.ndarray, encoder_hidden: np.ndarray):
    inps = decoder_sess.get_inputs()
    feed = {inps[0].name: input_ids, inps[1].name: encoder_hidden}
    outs = decoder_sess.run(None, feed)
    logits = outs[0]
    return logits

def greedy_decode_with_forced(encoder_sess, decoder_sess, input_features, processor, forced_decoder_ids, max_new_tokens=200):
    encoder_hidden = run_encoder(encoder_sess, input_features)

    if forced_decoder_ids.ndim == 1:
        input_ids = forced_decoder_ids[None].astype(np.int64)
    else:
        input_ids = forced_decoder_ids.astype(np.int64)

    generated = list(input_ids[0].tolist())

    for _ in range(max_new_tokens):
        logits = run_decoder_once(decoder_sess, np.array([generated], dtype=np.int64), encoder_hidden)
        next_logits = logits[:, -1, :]
        next_id = int(np.argmax(next_logits, axis=-1)[0])
        generated.append(next_id)
        if next_id == processor.tokenizer.eos_token_id:
            break

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()

# ==========================
# 2. Hàm đánh giá độ chính xác
# ==========================
def evaluate_folder(wav_folder: str, prompt_file: str, encoder_path: str, decoder_path: str, max_new_tokens=200):
    # Load model và processor
    print("🔹 Loading models & processor...")
    encoder_sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    decoder_sess = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
    processor = WhisperProcessor.from_pretrained("hkab/whisper-base-vietnamese-finetuned")

    forced_tok = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    forced_np = np.array(forced_tok, dtype=np.int64)
    if forced_np.ndim == 2 and forced_np.shape[0] > 1:
        forced_np = forced_np[0]

    # Load ground truth từ prompt.txt
    print("🔹 Loading ground truth from:", prompt_file)
    ground_truth = {}
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                name, text = line.strip().split("\t", 1)
            elif "|" in line:
                name, text = line.strip().split("|", 1)
            else:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                name, text = parts
            ground_truth[os.path.basename(name)] = text.strip()

    # breakpoint()

    wav_files = sorted(glob.glob(os.path.join(wav_folder, "*.wav")))
    if not wav_files:
        print("⚠️ Không tìm thấy file .wav trong thư mục:", wav_folder)
        return

    # Load metrics từ thư viện datasets
    print("🔹 Loading metrics...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    preds, refs, files = [], [], []

    # Inference từng file
    for wav in wav_files:
        basename = os.path.basename(wav)
        basename = os.path.splitext(basename)[0]
        # breakpoint()
        if basename not in ground_truth:
            print(f"⚠️ Không tìm thấy transcript cho {basename}, bỏ qua.")
            continue

        ref = ground_truth[basename]
        feats = extract_features(wav, processor)
        hyp = greedy_decode_with_forced(encoder_sess, decoder_sess, feats, processor, forced_np, max_new_tokens=max_new_tokens)

        preds.append(hyp)
        refs.append(ref)
        files.append(basename)

        print(f"[{len(preds)}/{len(wav_files)}] {basename}")
        print("   🔊 REF:", ref)
        print("   🤖 HYP:", hyp)

    # ==========================
    # 3. Tính chỉ số
    # ==========================
    print("\n🔹 Computing metrics...")
    wer_score = wer_metric.compute(predictions=preds, references=refs)
    cer_score = cer_metric.compute(predictions=preds, references=refs)

    print("\n===== 📊 Evaluation Results =====")
    print(f"WER : {wer_score:.4f}")
    print(f"CER : {cer_score:.4f}")


# ==========================
# 5. Chạy thực tế
# ==========================
if __name__ == "__main__":
    wav_folder = "/media/trandat/DataVoice/archive/vivos/test/waves/VIVOSDEV01"
    prompt_file = "/media/trandat/DataVoice/archive/vivos/test/prompts.txt"
    encoder_path = "whisper_onnx/encoder_model.onnx"
    decoder_path = "whisper_onnx/decoder_model.onnx"

    evaluate_folder(wav_folder, prompt_file, encoder_path, decoder_path)
