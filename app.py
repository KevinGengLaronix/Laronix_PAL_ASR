"""
TODO:
    + [x] Load Configuration
    + [ ] Multi ASR Engine
    + [ ] Batch / Real Time support
"""
import numpy as np
from pathlib import Path
import jiwer
import pdb
import torch.nn as nn
import torch
import torchaudio
import gradio as gr
from logging import PlaceHolder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2CTCTokenizer
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from datasets import load_dataset
import datasets
import yaml
from transformers import pipeline
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# local import
import sys

from local.vis import token_plot
sys.path.append("src")

# Load automos
config_yaml = "config/samples.yaml"
with open(config_yaml, "r") as f:
    # pdb.set_trace()
    try:
        config = yaml.safe_load(f)
    except FileExistsError:
        print("Config file Loading Error")
        exit()

# Auto load examples
refs = np.loadtxt(config["ref_txt"], delimiter="\n", dtype="str")
refs_ids = [x.split()[0] for x in refs]
refs_txt = [" ".join(x.split()[1:]) for x in refs]
ref_wavs = [str(x) for x in sorted(Path(config["ref_wavs"]).glob("**/*.wav"))]

# with open("src/description.html", "r", encoding="utf-8") as f:
#     description = f.read()
description = ""

reference_id = gr.Textbox(
    value="ID", placeholder="Utter ID", label="Reference_ID"
)
reference_textbox = gr.Textbox(
    value="Input reference here",
    placeholder="Input reference here",
    label="Reference",
)
reference_PPM = gr.Textbox(
    placeholder="Pneumatic Voice's PPM", label="Ref PPM"
)

examples = [
    [x, y] for x, y in zip(ref_wavs, refs_txt)
]

# def map_to_array(batch):
#     speech, _ = sf.read(batch["file"])
#     batch["speech"] = speech
#     return batch
# ASR part
p = pipeline("automatic-speech-recognition")
import pdb

# Tokenlizer part
# import model, feature extractor, tokenizer
def TOKENLIZER(audio_path, activate_plot=False):
    
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
    
    input_values, sr = torchaudio.load(audio_path)
    if sr != feature_extractor.sampling_rate:
        input_values = torchaudio.functional.resample(input_values, sr, feature_extractor.sampling_rate)

    logits = token_model(input_values).logits[0]
    pred_ids = torch.argmax(logits, axis=-1)

    # retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
    # pdb.set_trace()
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
    if activate_plot == True:
        token_fig = token_plot(input_values, feature_extractor.sampling_rate, word_offsets)
        return word_offsets, token_fig
    return word_offsets
# TOKENLIZER("data/samples/p326_020.wav")

# pdb.set_trace()
# Load dataset
# pdb.set_trace()
# dataset = load_dataset("common_voice", "en", split="train", streaming=True)
# dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
# dataset_iter = iter(dataset)
# sample = next(dataset_iter)

# pdb.set_trace()
# input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
# pdb.set_trace()

# WER part
transformation = jiwer.Compose(
    [
        jiwer.RemovePunctuation(),
        jiwer.ToUpperCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ]
)
()

class ChangeSampleRate(nn.Module):
    def __init__(self, input_rate: int, output_rate: int):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1)
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        indices = torch.arange(new_length) * (
            self.input_rate / self.output_rate
        )
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1.0 - indices.fmod(1.0)).unsqueeze(0) + (
            round_up * indices.fmod(1.0).unsqueeze(0)
        )
        return output

# Flagging setup

def calc_wer(audio_path, ref):
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] != 1:
        wav = wav[0, :].unsqueeze(0)
    print(wav.shape)
    osr = 16000
    batch = wav.unsqueeze(0).repeat(10, 1, 1)
    csr = ChangeSampleRate(sr, osr)
    out_wavs = csr(wav)
    # ASR
    # trans = jiwer.ToUpperCase()(p(audio_path)["text"])
    
    # Tokenlizer
    tokens, token_wav_plot = TOKENLIZER(audio_path, activate_plot=True)
    # ASR part
    
    trans_cnt = []
    for i in tokens:
        word, start_time, end_time = i.values()
        trans_cnt.append(word)
    trans = " ".join(x for x in trans_cnt)
    trans = jiwer.ToUpperCase()(trans)
    # WER
    ref = jiwer.ToUpperCase()(ref)
    wer = jiwer.wer(
        ref,
        trans,
        truth_transform=transformation,
        hypothesis_transform=transformation,
    )
    # pdb.set_trace()
    word_acc = 1.0 - float(wer)
    return [trans, word_acc, token_wav_plot]
# calc_wer(examples[1][0], examples[1][1])
# pdb.set_trace()
iface = gr.Interface(
    fn=calc_wer,
    inputs=[
        gr.Audio(
            source="upload",
            type="filepath",
            label="Audio_to_evaluate",
        ),
        reference_textbox,
    ],
    outputs=[
        gr.Textbox(placeholder="Hypothesis", label="Recognition by AI"),
        gr.Textbox(placeholder="Word Accuracy", label="Word Accuracy (The Higher the better)"),
        gr.Plot(label="waveform")
    ],
    description=description,
    examples=examples,
    examples_per_page=20,
    css=".body {background-color: green}",
)

print("Launch examples")

iface.launch(
    share=False,
)