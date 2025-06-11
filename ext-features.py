""" import torch, torchaudio, numpy as np, librosa, essentia.standard as es
from panns_inference import AudioTagging, labels as panns_labels
import json

wav_path = "/root/autodl-tmp/CLAP/data/classical_multi_dataset/25/4.wav"

# ---------- 0. 公用准备 ----------
# PANNs (527 标签概率)
model_sr = 32000                     # PANNs 模型采样率
tagger = AudioTagging(device="cpu")  # 或 "cuda"

wave, sr = torchaudio.load(wav_path)   # 原始采样率
if sr != model_sr:
    wave = torchaudio.functional.resample(wave, sr, model_sr)

clip_prob,frame_prob = tagger.inference(wave)  # 不再需要 tagger.resample


clip_prob = clip_prob[0]

# Librosa 22 kHz 单通道
y, sr = librosa.load(wav_path, sr=22050, mono=True)

# Essentia 16 kHz 单通道
audio16 = es.MonoLoader(filename=wav_path, sampleRate=16000)()

# ---------- 1. genre (离散) ----------
genre_groups = {
    "ambient": ["Ambient music", "New-age music"],
    "pop": ["Pop music"],
    "rock": ["Rock music", "Hard rock", "Alternative rock"],
    "classical": ["Classical music"],
    "jazz": ["Jazz", "Swing music"],
    "hip hop": ["Hip hop music", "Trap music"]
}
def top_genre(probs):
    score = {g: max(probs[panns_labels.index(t)] for t in tags if t in panns_labels)
             for g, tags in genre_groups.items()}
    return max(score, key=score.get)

genre = top_genre(clip_prob)

# ---------- 2. mood (离散) ----------
mood_keywords = ['happy', 'sad', 'angry', 'chill', 'calm', 'relax']

# 筛选包含上述关键词的标签
mood_tags = [
    tag for tag in panns_labels
    if any(kw in tag.lower() for kw in mood_keywords)
]

print("PANNs 内置的 mood 相关标签：")
for tag in mood_tags:
    print(tag)
mood_tags = [ "Happy music", "Sad music", "Angry music"]
mood = max(mood_tags, key=lambda t: clip_prob[panns_labels.index(t)])

# ---------- 3. danceability (连续 0‑1) ----------
dance = es.Danceability()(audio16)        # Essentia 预训练算法
primary = dance[0] if isinstance(dance, (tuple, list)) else dance
# 如果是数组就平均，否则直接转 float
dance_val = float(np.mean(primary)) if hasattr(primary, "__len__") else float(primary)
danceability = float(np.clip(dance_val, 0, 1))

# ---------- 4. energy (连续 0‑1) ----------
rms = librosa.feature.rms(y=y)
rms = float(rms.mean())  
print(f"  ▶ RMS value = {rms:.5f}") 
energy = float(np.clip((rms - 0.01) / 0.15, 0, 1))   # 经验归一化

# ---------- 5. acousticness (连续 0‑1) ----------
flatness = librosa.feature.spectral_flatness(y=y).mean()
acousticness = float(1 - flatness)                    # 越平坦越电子

# ---------- 6‑7. arousal / valence ->  valence ----------
with open('essentia_models/msd-musicnn-1.json') as f:
    emb_meta = json.load(f)

input_node = emb_meta['schema']['inputs'][0]['name']   # e.g. 'flatten_in_input'
# 从 outputs 里挑 shape == [1,200] 的那个 name：
output_node = next(
    o['name'] 
    for o in emb_meta['schema']['outputs'] 
    if o['shape'] == [1,200]
)

mood_cnn = es.TensorflowPredictMusiCNN(
    graphFilename='/root/autodl-tmp/CLAP/essentia_models/msd-musicnn-1.pb',
    patchSize=187,
    patchHopSize=93,
    output= output_node, 
    batchSize=0
)

embeddings = mood_cnn(audio16)

with open('/root/autodl-tmp/CLAP/essentia_models/deam-msd-musicnn-1.json') as f:
    metadata = json.load(f)
labels = metadata['classes']
av_labels = metadata['classes']       # ['arousal','valence']
head_input  = metadata['schema']['inputs'][0]['name']   # flatten_in_input
head_output = metadata['schema']['outputs'][0]['name']  # dense_out

av_head = es.TensorflowPredict2D(
    graphFilename='/root/autodl-tmp/CLAP/essentia_models/deam-msd-musicnn-1.pb',
    input=head_input,
    output=head_output,
    batchSize=0
)

activations = av_head(embeddings)   # shape (time, num_emotions)
mean_act    = np.mean(activations, axis=0)  # shape: (2,)
arousal = float(mean_act[av_labels.index('arousal')])
valence = float(mean_act[av_labels.index('valence')])
# ---------- 8. tempo (BPM) ----------
tempo = int(librosa.beat.tempo(y=y, sr=sr, max_tempo=200)[0])

# ---------- 汇总 ----------
features = dict(
    genre=genre,
    mood=mood.split()[0],        # "chill" / "happy" / …
    danceability=round(danceability, 2),
    energy=round(energy, 2),
    acousticness=round(acousticness, 2),
    valence=round(valence, 2),
    arousal=round(arousal,2),
    tempo=tempo
)

print(features)
# {'genre': 'ambient', 'mood': 'chill', 'danceability': 0.66,
#  'energy': 0.21, 'acousticness': 0.92, 'valence': 0.38, 'tempo': 75}
 """
# updated_extract_folder_features.py

# updated_extract_folder_features.py

# extract_all_folders_features.py

import os
import pandas as pd
from features_ext import (
    load_panns, load_librosa, load_essentia,
    get_genre, get_acousticness, get_energy,
    get_arousal_valence, get_tempo
)

def extract_features_in_folder(folder_path: str):
    """
    对 folder_path 下的所有 .wav 文件提取特征，
    并在同一文件夹内生成 <folder_name>.csv。
    'file' 列仅保留文件名的数字部分（0,1,2,...），按数值排序。
    """
    # 列出所有 .wav 文件
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    if not files:
        return
    # 按文件名中的数字排序
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    records = []
    for fn in files:
        base = os.path.splitext(fn)[0]
        path = os.path.join(folder_path, fn)
        clip_prob = load_panns(path)
        genre = get_genre(clip_prob)
        y = load_librosa(path)
        audio16 = load_essentia(path)
        acousticness = get_acousticness(y)
        acousticness=round(acousticness,4)
        energy       = get_energy(y)
        energy=round(energy,4)
        arousal, valence = get_arousal_valence(audio16)
        arousal=round(arousal,4)
        valence=round(valence,4)
        tempo        = get_tempo(y)
        records.append({
            'file': int(base),
            'genre': genre,
            'acousticness': acousticness,
            'energy': energy,
            'valence': valence,
            'arousal': arousal,
            'tempo': tempo
        })

    df = pd.DataFrame(records).sort_values('file')
    folder_name = os.path.basename(os.path.normpath(folder_path))
    out_csv = os.path.join(folder_path, f"{folder_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved features to {out_csv}")

def extract_all(parent_dir: str):
    """
    如果 parent_dir 本身包含 .wav 文件，则直接提取；
    否则，对 parent_dir 下每个子文件夹执行提取。
    """
    # 如果当前目录有 wav 文件，处理自己
    wavs = [f for f in os.listdir(parent_dir) if f.lower().endswith('.wav')]
    if wavs:
        extract_features_in_folder(parent_dir)
    else:
        # 否则遍历子目录
        for name in sorted(os.listdir(parent_dir), key=lambda x:(0,int(x)) if x.isdigit() else (1,x)):
            sub = os.path.join(parent_dir, name)
            if os.path.isdir(sub):
                extract_features_in_folder(sub)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_all_folders_features.py <parent_directory>")
    else:
        extract_all(sys.argv[1])

