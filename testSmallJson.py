from pathlib import Path
import os
import json
import librosa

input_dir = Path.cwd() / "ConvertedFiles"

transcript_dir = Path.cwd() / "Transcripts"

files = list(input_dir.rglob("*.wav*"))

transcripts = list(transcript_dir.rglob("*.txt*"))

transcriptData = ""

datas = []
        
for path in files:
    for path2 in transcripts:
        if path.parent.name in os.path.basename(path2):
            file = open(path2, 'r')
            f = file.readlines()
            for line in f:
                if line.split(' ')[0] in os.path.basename(path):
                    transcriptData = line.split(" ", 1)[1].strip()
                    print(transcriptData)
    data = {}
    data = {
            "Name":os.path.basename(path),
            "Path":os.path.abspath(path),
            "Transcript": transcriptData,
    }
    datas.append(data)

with open("test.json", "w") as f:
    json.dump(datas, f, indent = 4)
    print("Done")