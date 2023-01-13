import numpy as np
from pathlib import Path
import jiwer
import pdb
import torch.nn as nn
import torch
import torchaudio
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC

import yaml
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf


def TOKENLIZER(audio_path):
    token_model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

    # # load first sample of English common_voice
    # dataset = load_dataset("common_voice", "en", split="train", streaming=True)
    # dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    # dataset_iter = iter(dataset)
    # sample = next(dataset_iter)

    # # forward sample through model to get greedily predicted transcription ids
    # input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
    # pdb.set_trace()
    
    # load samples
    input_values, sr = torchaudio.load(audio_path)
    # resample
    if sr != feature_extractor.sampling_rate:
        input_values = torchaudio.functional.resample(input_values, sr, feature_extractor.sampling_rate)

    logits = token_model(input_values).logits[0]
    # Get predict IDs
    pred_ids = torch.argmax(logits, axis=-1)

    # retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = token_model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

    word_offsets = [
        {
            "word": d["word"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
        }
        for d in outputs.word_offsets
    ]
    return word_offsets