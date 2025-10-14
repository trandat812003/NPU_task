import numpy as np
import onnxruntime as ort
import librosa
from transformers import WhisperProcessor


def extract_features(wav_path: str, processor: WhisperProcessor):
    audio, sr = librosa.load(wav_path, sr=16000)
    feats = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="np")
    return feats["input_features"]


def run_encoder(encoder_sess: ort.InferenceSession, input_features: np.ndarray):
    inp = input_features.astype(np.float32)
    name = encoder_sess.get_inputs()[0].name
    outs = encoder_sess.run(None, {name: inp})
    return outs[0]


def run_decoder_once(
    decoder_sess: ort.InferenceSession,
    input_ids: np.ndarray,
    encoder_hidden: np.ndarray,
):
    inps = decoder_sess.get_inputs()
    feed = {inps[0].name: input_ids, inps[1].name: encoder_hidden}
    outs = decoder_sess.run(None, feed)
    # outs[0]: logits, outs[1:]: present.*
    logits = outs[0]
    presents = outs[1:]
    return logits, presents


def greedy_decode_with_forced(
    encoder_sess,
    decoder_sess,
    input_features: np.ndarray,
    processor: WhisperProcessor,
    forced_decoder_ids: np.ndarray,
    max_new_tokens=100,
):
    """
    forced_decoder_ids: numpy array of shape (batch, prefix_length)
    """
    encoder_hidden = run_encoder(encoder_sess, input_features)

    # start token ids: forced prefix
    # forced_decoder_ids is a sequence of tokens you want to start with
    # We will feed that prefix as input_ids first, then decode from there
    input_ids = forced_decoder_ids.copy().astype(np.int64)  # shape (1, prefix_len)
    generated = list(input_ids[0])  # copy prefix tokens into generated list

    for _ in range(max_new_tokens):
        logits, presents = run_decoder_once(decoder_sess, input_ids, encoder_hidden)
        next_logits = logits[:, -1, :]  # shape (batch, vocab_size)
        next_id = int(np.argmax(next_logits, axis=-1)[0])

        generated.append(next_id)
        if next_id == processor.tokenizer.eos_token_id:
            break

        # Now feed entire generated sequence (no cache version)
        input_ids = np.array([generated], dtype=np.int64)

    # Decode full generated sequence, but skip the prefix (forced) if needed
    # or use skip_special_tokens
    # Using `processor.tokenizer.decode(...)`
    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return text


def main():
    encoder_path = "whisper_onnx/encoder_model.onnx"
    decoder_path = "whisper_onnx/decoder_model.onnx"

    encoder_sess = ort.InferenceSession(
        encoder_path, providers=["CPUExecutionProvider"]
    )
    decoder_sess = ort.InferenceSession(
        decoder_path, providers=["CPUExecutionProvider"]
    )

    processor = WhisperProcessor.from_pretrained(
        "hkab/whisper-base-vietnamese-finetuned"
    )

    # get forced_decoder_ids as in PyTorch version
    # forced_decoder_ids is a torch tensor, convert to numpy
    forced_tok = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    # forced_tok is a list of lists or torch tensor — convert:
    forced_np = np.array(forced_tok, dtype=np.int64)
    breakpoint()

    wav = "/home/trandat/Documents/vnpt/NPU_task/demo.wav"
    feats = extract_features(wav, processor)  # shape (1, 80, seq_len)

    transcript = greedy_decode_with_forced(
        encoder_sess, decoder_sess, feats, processor, forced_np, max_new_tokens=200
    )
    print("Prediction ONNX:", transcript)


if __name__ == "__main__":
    main()
