# Code heavily based on code from https://huggingface.co/openai/whisper-large-v3

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from scipy.io import wavfile

device = "cuda"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(device)
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)

model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

import pandas

dataset = pandas.read_json("test.json")

from tqdm import tqdm
import librosa
from scipy.io import wavfile
import wave

predictions = []
taehoon_references = []
shawne_references = []
count = 1

file1 = open("transcriptions.txt", "w")

# Get and store transcriptions for whisper for each audio file
# Also stores transcriptions from Shawne and Taehoon
for ind in dataset.index:
    print(count)
    soundFile = dataset['Path'][ind]
    
    signal, sr = librosa.load(soundFile)
    
    result = pipe(signal, generate_kwargs={"language":"english"})

    predictions.append(processor.tokenizer._normalize(result['text']))
    taehoon_references.append(processor.tokenizer._normalize(dataset['Taehoon'][ind]))
    shawne_references.append(processor.tokenizer._normalize(dataset['Shawne'][ind]))
    count = count+1

    
    file1.write(result['text'] + "\n\n")


file1.close()

import evaluate
from evaluate import load

wer = load("wer")
print(1.0-wer.compute(predictions=predictions, references=taehoon_references))
print(1.0-wer.compute(predictions=predictions, references=shawne_references))
