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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import yaml
from transformers import pipeline
import librosa
import librosa.display
import matplotlib.pyplot as plt

# local import
import sys

sys.path.append("src")
import lightning_module

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

with open("src/description.html", "r", encoding="utf-8") as f:
    description = f.read()
# description

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

# ASR part
p = pipeline("automatic-speech-recognition")

# WER part
transformation = jiwer.Compose(
    [
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ]
)

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
        wav = wav[0, :]
    print(wav.shape)
    
    osr = 16000
    batch = wav.unsqueeze(0).repeat(10, 1, 1)
    csr = ChangeSampleRate(sr, osr)
    out_wavs = csr(wav)
    # ASR
    trans = jiwer.ToLowerCase()(p(audio_path)["text"])

    # WER
    wer = jiwer.wer(
        ref,
        trans,
        truth_transform=transformation,
        hypothesis_transform=transformation,
    )
    
    return [trans, wer]
    
iface = gr.Interface(
    fn=calc_wer,
    inputs=[
        gr.Audio(
            source="microphone",
            type="filepath",
            label="Audio_to_evaluate",
        ),
        reference_textbox
    ],
    outputs=[
        gr.Textbox(placeholder="Hypothesis", label="Hypothesis"),
        gr.Textbox(placeholder="Word Error Rate", label="WER"),
    ],
    title="Laronix Automatic Speech Recognition",
    description=description,
    examples=examples,
    css=".body {background-color: green}",
)

print("Launch examples")

iface.launch(
    share=False,
)