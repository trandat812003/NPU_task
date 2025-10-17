# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os

import numpy as np
import samplerate
import torch
import whisper  
from scipy import special as scipy_special  
import audio2numpy as a2n
import argparse

from qai_hub_models.models._shared.whisper.model import (
    CHUNK_LENGTH,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    MEAN_DECODE_LEN,
    AUDIO_EMB_LEN,
)
from qai_hub_models.models._shared.whisper.model import (
    CollectionModel,
    BaseModel,
    SplitLinear,
    ResidualAttentionBlockWrapper,
    Whisper,
    WhisperDecoderInf,
    WhisperEncoderInf,
    TargetRuntime,
    Precision
)

from qai_appbuilder import (QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig)
from typing import Any, Optional
from qai_hub.client import Device
from qai_hub_models.utils.input_spec import InputSpec
from transformers import WhisperProcessor



ENCODER_MODEL_NAME = "whisper-encoder"
DECODER_MODEL_NAME = "whisper-decoder"
WHISPER_VERSION = "base" #"base.en"


execution_ws = os.getcwd()
qnn_dir = execution_ws + "\\qai_libs"

model_dir = execution_ws + "\\models"
encoder_model_path = model_dir + "\\" + ENCODER_MODEL_NAME + ".bin"
decoder_model_path = model_dir + "\\" + DECODER_MODEL_NAME + ".bin"

mel_filter_path = execution_ws + "\\mel_filters.npz"


# Whisper constants
TOKEN_SOT = 50257  # Start of transcript
TOKEN_EOT = 50256  # end of transcript
TOKEN_BLANK = 220  # " "
TOKEN_NO_TIMESTAMP = 50362
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH = 50361
SAMPLE_RATE = 16000

MAX_AUDIO_SAMPLES=CHUNK_LENGTH * SAMPLE_RATE #

# Above this prob we deem there's no speech in the audio
NO_SPEECH_THR = 0.6

# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L600
NON_SPEECH_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    357,
    366,
    438,
    532,
    685,
    705,
    796,
    930,
    1058,
    1220,
    1267,
    1279,
    1303,
    1343,
    1377,
    1391,
    1635,
    1782,
    1875,
    2162,
    2361,
    2488,
    3467,
    4008,
    4211,
    4600,
    4808,
    5299,
    5855,
    6329,
    7203,
    9609,
    9959,
    10563,
    10786,
    11420,
    11709,
    11907,
    13163,
    13697,
    13700,
    14808,
    15306,
    16410,
    16791,
    17992,
    19203,
    19510,
    20724,
    22305,
    22935,
    27007,
    30109,
    30420,
    33409,
    34949,
    40283,
    40493,
    40549,
    47282,
    49146,
    50257,
    50357,
    50358,
    50359,
    50360,
    50361,
]

SAMPLE_BEGIN = 1  # first token is TOKEN_SOT


# https://github.com/openai/whisper/blob/v20230314/whisper/decoding.py#L545
precision = 0.02  # in second
max_initial_timestamp = 1.0  # in second
max_initial_timestamp_index = int(max_initial_timestamp / precision)


encoder=None
decoder=None
whisper_base_en=None
mel_filter=None

class WhisperDecoderInf(BaseModel):
    """
    whisper.model.TextDecoder optimized for export and inference:

    Wraps `whisper.model.TextDecoder` to facilitate export:

    1. kv cache inputs are individual tensors instead of a list of tensors
    2. kv cache inputs are required, not optional
    """

    def __init__(
        self, model, max_decode_len: int = MEAN_DECODE_LEN
    ):
        super().__init__()
        model = model.decoder

        self.max_decode_len = max_decode_len

        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlockWrapper(b) for b in model.blocks]
        )

        for m in ["token_embedding", "ln"]:
            self.add_module(m, getattr(model, m))

        self.positional_embedding = torch.nn.Embedding(
            max_decode_len, self.token_embedding.weight.shape[1]
        )
        self.positional_embedding.weight = torch.nn.Parameter(
            model.positional_embedding.weight[:max_decode_len, :]
        )

        self.logits = torch.nn.Linear(
            self.token_embedding.weight.shape[1],
            self.token_embedding.weight.shape[0],
            bias=False,
        )
        self.logits.weight = self.token_embedding.weight

        self.mask = torch.nn.Embedding(max_decode_len, max_decode_len)
        mask = torch.zeros([max_decode_len, max_decode_len], dtype=torch.float32)
        for c_idx in range(0, max_decode_len):
            mask[c_idx, 0 : max_decode_len - c_idx - 1] = -100
        self.mask.weight = torch.nn.Parameter(mask)

    @property
    def attention_dim(self):
        return self.blocks[0].attn_ln.weight.shape[0]

    @property
    def num_heads(self):
        return self.blocks[0].attn.n_head

    @property
    def num_blocks(self):
        return len(self.blocks)

    def forward(
        self,
        x: torch.Tensor,
        index: torch.Tensor,
        k_cache_cross: torch.Tensor,
        v_cache_cross: torch.Tensor,
        k_cache_self: torch.Tensor,
        v_cache_self: torch.Tensor,
    ):
        """
        Args:

        - x: torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens

        - index: torch.tensor, shape = (1, 1)
            index to get the positional encoding for x.

        - k_cache_cross: key cache for cross attention:
          [num_blocks, num_heads, attn_dim/num_heads, AUDIO_EMB_LEN]

        - v_cache_cross: value cache for cross attention:
          [num_blocks, num_heads, AUDIO_EMB_LEN, attn_dim/num_heads]

        - k_cache_self: key cache for self attention:
          [num_blocks, num_heads, attn_dim/num_heads, self.max_decode_len]
          pass zeros for first call (index 0), otherwise pass in
          previous decoder output

        - v_cache_self: value cache for self attention:
          [num_blocks, num_heads, self.max_decode_len, attn_dim/num_heads]
          pass zeros for first call (index 0), otherwise pass in
          previous decoder output

        Returns:

        - logits: of shape [1, 1, 51864]
        - k_cache_self_new: updated key cache for self attention
        - v_cache_self_new: updated value cache for self attention
        """

        assert isinstance(self.token_embedding, torch.nn.Module)  # for mypy
        assert isinstance(self.ln, torch.nn.Module)  # for mypy
        assert isinstance(self.positional_embedding, torch.nn.Embedding)  # for mypy
        # Set up kv_cache
        kv_cache = {}  # torch.nn.Module -> torch.Tensor
        for i, block in enumerate(self.blocks):
            kv_cache.update(
                {
                    block.attn.key: k_cache_self[i : i + 1],
                    block.attn.value: v_cache_self[i : i + 1],
                    block.cross_attn.key: k_cache_cross[i : i + 1],
                    block.cross_attn.value: v_cache_cross[i : i + 1],
                }
            )

        x = self.token_embedding(x) + self.positional_embedding(index)
        mask = self.mask(index)

        # x shape: (1, 1, 384)
        k_cache_new = []
        v_cache_new = []
        for block_idx in range(self.num_blocks):
            x, k_cache, v_cache = self.blocks[block_idx](x, mask, kv_cache=kv_cache)
            k_cache_new.append(k_cache.float())
            v_cache_new.append(v_cache.float())

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        logits = self.logits(x).float()

        return logits, torch.cat(k_cache_new), torch.cat(v_cache_new)

    @staticmethod
    def get_input_spec(
        num_blocks: int, attention_dim: int, num_heads: int
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        specs: InputSpec = dict(
            x=((1, 1), "int32"),
            index=((1, 1), "int32"),
            k_cache_cross=(
                (num_blocks, num_heads, attention_dim // num_heads, AUDIO_EMB_LEN),
                "float32",
            ),
            v_cache_cross=(
                (num_blocks, num_heads, AUDIO_EMB_LEN, attention_dim // num_heads),
                "float32",
            ),
            k_cache_self=(
                (num_blocks, num_heads, attention_dim // num_heads, MEAN_DECODE_LEN),
                "float32",
            ),
            v_cache_self=(
                (num_blocks, num_heads, MEAN_DECODE_LEN, attention_dim // num_heads),
                "float32",
            ),
        )

        return specs

    @staticmethod
    def get_output_names() -> list[str]:
        return ["logits", "k_cache", "v_cache"]

    def _get_input_spec_for_instance(self) -> InputSpec:
        return self.__class__.get_input_spec(
            len(self.blocks), self.attention_dim, self.num_heads
        )

    @classmethod
    def from_pretrained(cls):
        return Whisper.from_pretrained().decoder

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Optional[Device] = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if precision == Precision.float and target_runtime in {
            TargetRuntime.QNN,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        }:
            compile_options = (
                compile_options + " --quantize_full_type float16 --quantize_io"
            )
        return compile_options

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class Encoder(QNNContext):
    def __init__(self,
                model: whisper.model.Whisper,  
                model_name: str = "None",
                model_path: str = "None",
                backend_lib_path: str = "None",
                system_lib_path: str = "None",
                runtime: str = Runtime.HTP,
                is_async: bool = False,
    ):
        super().__init__(model_name, model_path, backend_lib_path, system_lib_path, runtime, is_async)
        self.encoder = model.encoder
        n_audio_state=512
        n_audio_head=8

        states_per_head = n_audio_state // n_audio_head
        scale = states_per_head**-0.25

        self.cross_attn_key_list = torch.nn.ModuleList(
            [
                SplitLinear(block.cross_attn.key, n_audio_head, scale)
                for block in model.decoder.blocks
            ]
        )
        self.cross_attn_value_list = torch.nn.ModuleList(
            [
                SplitLinear(block.cross_attn.value, n_audio_head)
                for block in model.decoder.blocks
            ]
        )

    def Inference(self, input_data):
        input_datas=[input_data]
        output_data = super().Inference(input_datas) 
        output_data = output_data[0].reshape(1, 1500, 512) 
        output_data = torch.Tensor(output_data)
        k_cache = torch.cat(
            [
                key(output_data, transpose=True).unsqueeze(0)
                for key in self.cross_attn_key_list
            ],
            dim=0,
        )
        v_cache = torch.cat(
            [value(output_data).unsqueeze(0) for value in self.cross_attn_value_list],
            dim=0,
        )
        return k_cache, v_cache
        
class Decoder(QNNContext, WhisperDecoderInf):
    def __init__(self,
                model: whisper.model.Whisper,  
                model_name: str = "None",
                model_path: str = "None",
                backend_lib_path: str = "None",
                system_lib_path: str = "None",
                runtime: str = Runtime.HTP,
                is_async: bool = False,
                max_decode_len: int = MEAN_DECODE_LEN,
    ):
        QNNContext.__init__(self,
                            model_name=model_name,
                            model_path=model_path,
                            backend_lib_path=backend_lib_path,
                            system_lib_path=system_lib_path,
                            runtime=runtime,
                            is_async=is_async)
        
        WhisperDecoderInf.__init__(self, model=model, max_decode_len=max_decode_len)

    def Inference(self, 
                  x: torch.Tensor, 
                  index: torch.Tensor, 
                  k_cache_cross: torch.Tensor, 
                  v_cache_cross: torch.Tensor, 
                  k_cache_self: torch.Tensor, 
                  v_cache_self: torch.Tensor,
    ):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).long()
        if isinstance(index, np.ndarray):
            index = torch.from_numpy(index).long()
        if isinstance(k_cache_cross, np.ndarray):
            k_cache_cross = torch.from_numpy(k_cache_cross).float()
        if isinstance(v_cache_cross, np.ndarray):
            v_cache_cross = torch.from_numpy(v_cache_cross).float()
        if isinstance(k_cache_self, np.ndarray):
            k_cache_self = torch.from_numpy(k_cache_self).float()
        if isinstance(v_cache_self, np.ndarray):
            v_cache_self = torch.from_numpy(v_cache_self).float()
        return self.forward(x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self)


@CollectionModel.add_component(WhisperEncoderInf)
@CollectionModel.add_component(WhisperDecoderInf)
class WhisperBaseEn(Whisper):
    @classmethod
    def from_pretrained(cls):
        return super().from_pretrained(r"C:\Users\asus\Documents\datnt\NPU_task\whisper.pt") 

def Init():
    global encoder,decoder,whisper_base_en,mel_filter

    with np.load(mel_filter_path) as f:
       mel_filter = f[f"mel_{N_MELS}"]

    whisper_base_en =WhisperBaseEn.from_pretrained() 

    # Config AppBuilder environment.
    QNNConfig.Config(qnn_dir, Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)

    decoder = Decoder(whisper_base_en, "whisper_decoder", decoder_model_path)
    encoder = Encoder(whisper_base_en, "whisper_encoder", encoder_model_path)

def Inference(audio_path):
    audio, audio_sample_rate = a2n.audio_from_file(audio_path)

    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

    result=" ".join(
            transcribe_single_chunk(x)
            for x in chunk_and_resample_audio(audio, audio_sample_rate)
        )

    PerfProfile.RelPerfProfileGlobal()
    
    print("Transcription:",result)
    

def transcribe_single_chunk(audio: np.ndarray):
    mel_input = log_mel_spectrogram(
            mel_filter, audio, MAX_AUDIO_SAMPLES, N_FFT, HOP_LENGTH
    )
    k_cache_cross = np.zeros(
        (
            6,
            8,
            64,
            1500,
        )
    ).astype(np.float32)
    v_cache_cross = np.zeros(
        (
            6,
            8,
            1500,
            64,
        )
    ).astype(np.float32)
    k_cache_cross, v_cache_cross = encoder.Inference(mel_input)
    x = np.array([[TOKEN_SOT]])
    decoded_tokens = [TOKEN_SOT]
    sample_len = whisper_base_en.mean_decode_len

    logits = np.zeros(
        (
            1,
            1,
            51864,
        )
    ).astype(np.float32)
    k_cache_self = np.zeros(
        (
            6,
            8,
            64,
            224,
        )
    ).astype(np.float32)
    v_cache_self = np.zeros(
        (
            6,
            8,
            224,
            64,
        )
    ).astype(np.float32)
        
    sum_logprobs = 0
    for i in range(sample_len):
        index = torch.zeros([1, 1], dtype=torch.int32)
        index[0, 0] = i
        logits, k_cache_self, v_cache_self = decoder.Inference(
            x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self
        )
        logits = logits[0, -1] 
        logits = logits.detach().cpu().numpy()
        if i == 0:
            logits[[TOKEN_EOT, TOKEN_BLANK]] = -np.inf
        logits[NON_SPEECH_TOKENS] = -np.inf

        logits, logprobs = apply_timestamp_rules(logits, decoded_tokens)
        assert isinstance(logprobs, np.ndarray)

        if i == 0:
            no_speech_prob = np.exp(logprobs[TOKEN_NO_SPEECH])
            if no_speech_prob > NO_SPEECH_THR:
                break

        next_token = np.argmax(logits)
        if next_token == TOKEN_EOT:
            break

        sum_logprobs += logprobs[next_token]
        x = np.array([[next_token]])
        decoded_tokens.append(int(next_token))

    processor = WhisperProcessor.from_pretrained("hkab/whisper-base-vietnamese-finetuned")
    text = processor.tokenizer.decode(decoded_tokens, skip_special_tokens=True)

    # text = tokenizer.decode(decoded_tokens[1:])
    return text.strip()


def apply_timestamp_rules(
    logits: np.ndarray, tokens: list[int]
) -> tuple[np.ndarray, float | np.ndarray]:
    # Require producing timestamp
    logits[TOKEN_NO_TIMESTAMP] = -np.inf

    # timestamps have to appear in pairs, except directly before EOT
    seq = tokens[SAMPLE_BEGIN:]
    last_was_timestamp = len(seq) >= 1 and seq[-1] >= TOKEN_TIMESTAMP_BEGIN
    penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= TOKEN_TIMESTAMP_BEGIN
    if last_was_timestamp:
        if penultimate_was_timestamp:  # has to be non-timestamp
            logits[TOKEN_TIMESTAMP_BEGIN:] = -np.inf
        else:  # cannot be normal text tokens
            logits[:TOKEN_EOT] = -np.inf

    timestamps = [t for t in tokens if t >= TOKEN_TIMESTAMP_BEGIN]
    if len(timestamps) > 0:
        # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
        # also force each segment to have a nonzero length, to   prevent infinite looping
        if last_was_timestamp and not penultimate_was_timestamp:
            timestamp_last = timestamps[-1]
        else:
            timestamp_last = timestamps[-1] + 1
        logits[TOKEN_TIMESTAMP_BEGIN:timestamp_last] = -np.inf

    if len(tokens) == SAMPLE_BEGIN:
        # suppress generating non-timestamp tokens at the beginning
        logits[:TOKEN_TIMESTAMP_BEGIN] = -np.inf

        # apply the `max_initial_timestamp` option
        last_allowed = TOKEN_TIMESTAMP_BEGIN + max_initial_timestamp_index
        logits[(last_allowed + 1) :] = -np.inf

    # if sum of probability over timestamps is above any other token, sample timestamp
    logprobs = scipy_special.log_softmax(logits)
    timestamp_logprob = scipy_special.logsumexp(logprobs[TOKEN_TIMESTAMP_BEGIN:])
    max_text_token_logprob = logprobs[:TOKEN_TIMESTAMP_BEGIN].max()
    if timestamp_logprob > max_text_token_logprob:
        # Mask out all but timestamp tokens
        logits[:TOKEN_TIMESTAMP_BEGIN] = -np.inf

    return logits, logprobs


# Adopted from https://github.com/openai/whisper/blob/main/whisper/audio.py
def log_mel_spectrogram(
    mel_filter: np.ndarray,
    audio_np: np.ndarray,
    pad_to_length: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    audio = torch.from_numpy(audio_np)
    assert isinstance(audio, torch.Tensor)

    if pad_to_length is not None:
        padding = pad_to_length - len(audio)
        if padding > 0:
            audio = torch.nn.functional.pad(audio, (0, padding))
    window = torch.hann_window(n_fft)
    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    mel_spec = torch.from_numpy(mel_filter) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.unsqueeze(0).detach().float().numpy()


def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate=SAMPLE_RATE,
    model_chunk_seconds=CHUNK_LENGTH,
) -> list[np.ndarray]:
    if audio_sample_rate != model_sample_rate:
        audio = samplerate.resample(audio, model_sample_rate / audio_sample_rate)
        audio_sample_rate = model_sample_rate

    number_of_full_length_audio_chunks = (
        audio.shape[0] // audio_sample_rate // model_chunk_seconds
    )
    last_sample_in_full_length_audio_chunks = (
        audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
    )

    if number_of_full_length_audio_chunks == 0:
        return [audio]

    return [
        *np.array_split(
            audio[:last_sample_in_full_length_audio_chunks],
            number_of_full_length_audio_chunks,
        ),
        audio[last_sample_in_full_length_audio_chunks:],
    ]


def Release():
    global decoder,encoder,whisper_base_en

    # Release the resources.
    del(decoder)
    del(encoder)
    del(whisper_base_en)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file",
        type=str,
        default=execution_ws+"\\demo.wav",
        help="Audio file path ",
    )
    args = parser.parse_args()

    Init()

    Inference(args.audio_file)

    Release()
    

if __name__ == '__main__':
    main()
