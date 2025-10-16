import os
import numpy as np
import torch
import samplerate
import time
import argparse
import audio2numpy as a2n
from scipy import special as scipy_special
from transformers import WhisperProcessor
from qai_appbuilder import QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig


execution_ws = os.getcwd()
qnn_dir = os.path.join(execution_ws, "demo_qualcomm", "qai_libs")

encoder_model_path = os.path.join(execution_ws, "whisper-encoder.bin")
decoder_model_path = os.path.join(execution_ws, "whisper-decoder.bin")
SAMPLE_RATE = 16000


TOKEN_SOT = 50257
TOKEN_EOT = 50256


class Encoder(QNNContext):
    def Inference(self, input_features):
        out = super().Inference([input_features])
        out = out[0]
        # if out.ndim == 1:
        #     out = out.reshape(1, 1500, 512)
        # elif out.ndim == 2 and out.shape[-1] == 512:
        #     out = out[None, :, :] 

        return out

class Decoder(QNNContext):
    def Inference(self, input_ids, encoder_hidden_states):
        outs = super().Inference([input_ids, encoder_hidden_states])
        return outs[0]

def extract_features(wav_path: str, processor: WhisperProcessor):
    audio, sr = a2n.audio_from_file(wav_path)
    if sr != SAMPLE_RATE:
        audio = samplerate.resample(audio, SAMPLE_RATE / sr)
    feats = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    return feats["input_features"].astype(np.float32)

def run_encoder(encoder_sess, input_features: np.ndarray):
    outs = encoder_sess.Inference(input_features)
    return outs

def run_decoder_once(decoder_sess, input_ids: np.ndarray, encoder_hidden: np.ndarray):
    breakpoint()
    outs = decoder_sess.Inference(input_ids, encoder_hidden)
    logits = outs
    return logits

def greedy_decode_with_forced(encoder_sess, decoder_sess, input_features: np.ndarray, processor: WhisperProcessor, forced_decoder_ids: np.ndarray, max_new_tokens=200):
    encoder_hidden = run_encoder(encoder_sess, input_features)

    if forced_decoder_ids.ndim == 1:
        input_ids = forced_decoder_ids[None].astype(np.int64)
    else:
        input_ids = forced_decoder_ids.astype(np.int64)

    generated = list(input_ids[0].tolist())

    breakpoint()

    for _ in range(max_new_tokens):
        logits = run_decoder_once(decoder_sess, np.array([generated], dtype=np.int64), encoder_hidden)
        breakpoint()
        next_logits = logits[:, -1, :]  # (batch, vocab)
        next_id = int(np.argmax(next_logits, axis=-1)[0])
        generated.append(next_id)
        if next_id == processor.tokenizer.eos_token_id:
            break

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return text

# --------------------------------------------------------------
# Pipeline chính
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", type=str, default="demo.wav", help="Audio input path")
    args = parser.parse_args()

    # Cấu hình môi trường QNN
    QNNConfig.Config(qnn_dir, Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

    print("[1] Load models .bin ...")
    encoder = Encoder("whisper_encoder", encoder_model_path)
    decoder = Decoder("whisper_decoder", decoder_model_path)

    print("[2] Load processor ...")
    processor = WhisperProcessor.from_pretrained("hkab/whisper-base-vietnamese-finetuned")

    print("[3] Extract features ...")
    feats = extract_features(args.audio_file, processor)

    forced_tok = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    forced_np = np.array(forced_tok, dtype=np.int64)
    if forced_np.ndim == 2 and forced_np.shape[0] > 1:
        forced_np = forced_np[0]

    print("[4] Start decoding ...")
    t0 = time.time()
    transcript = greedy_decode_with_forced(encoder, decoder, feats, processor, forced_np, max_new_tokens=200)
    t1 = time.time()

    print(f"[DONE] Time: {t1 - t0:.2f}s")
    print("Prediction:", transcript)

    # Giải phóng tài nguyên
    del encoder, decoder
    PerfProfile.RelPerfProfileGlobal()

if __name__ == "__main__":
    main()
