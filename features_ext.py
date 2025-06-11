# feature_extractor.py

import torch
import torchaudio
import numpy as np
import librosa
import essentia.standard as es
from panns_inference import AudioTagging, labels as panns_labels
import json

def load_panns(wav_path: str, device: str = "cpu", model_sr: int = 32000) -> np.ndarray:
    """Load audio and run PANNs inference, returning clip-level probabilities."""
    tagger = AudioTagging(device=device)
    wave, sr = torchaudio.load(wav_path)
    if sr != model_sr:
        wave = torchaudio.functional.resample(wave, sr, model_sr)
    clip_prob, _ = tagger.inference(wave)
    return clip_prob[0]

def load_librosa(wav_path: str, sr: int = 22050) -> np.ndarray:
    """Load audio with librosa at target SR, mono."""
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    return y

def load_essentia(wav_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio with Essentia's MonoLoader."""
    return es.MonoLoader(filename=wav_path, sampleRate=sr)()

def get_genre(clip_prob: np.ndarray) -> str:
    """Map PANNs clip_prob to predefined broad genres."""
    genre_groups = {
        "ambient": ["Ambient music", "New-age music"],
        "pop": ["Pop music"],
        "rock": ["Rock music", "Hard rock", "Alternative rock"],
        "classical": ["Classical music"],
        "jazz": ["Jazz", "Swing music"],
        "hip hop": ["Hip hop music", "Trap music"]
    }
    scores = {
        g: max(clip_prob[panns_labels.index(t)] for t in tags if t in panns_labels)
        for g, tags in genre_groups.items()
    }
    return max(scores, key=scores.get)

def get_mood(clip_prob: np.ndarray, mood_tags=None) -> str:
    """Select a single mood from PANNs tags."""
    if mood_tags is None:
        mood_tags = ["Happy music", "Sad music", "Angry music"]
    return max(mood_tags, key=lambda t: clip_prob[panns_labels.index(t)])

def get_danceability(audio16: np.ndarray) -> float:
    """Compute danceability via Essentia's Danceability."""
    raw = es.Danceability()(audio16)
    primary = raw[0] if isinstance(raw, (tuple, list)) else raw
    arr = np.asarray(primary).flatten()
    return float(np.clip(arr.mean(), 0, 1))

def get_energy(y: np.ndarray) -> float:
    """Compute RMS-based energy [0,1]."""
    rms_vals = librosa.feature.rms(y=y)
    rms = float(rms_vals.mean())
    return float(np.clip((rms - 0.01) / 0.15, 0, 1))

def get_acousticness(y: np.ndarray) -> float:
    """Compute acousticness as 1 - spectral_flatness."""
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    return float(np.clip(1 - flatness, 0, 1))

def get_arousal_valence(audio16: np.ndarray,
                        emb_pb: str = 'essentia_models/msd-musicnn-1.pb',
                        emb_json: str = 'essentia_models/msd-musicnn-1.json',
                        head_pb: str = 'essentia_models/deam-msd-musicnn-1.pb',
                        head_json: str = 'essentia_models/deam-msd-musicnn-1.json',
                        required_patches=6) -> tuple:
    """Compute arousal and valence via Essentia DEAM head."""
    # Embedding
    with open(emb_json) as f:
        emb_meta = json.load(f)
    inp = emb_meta['schema']['inputs'][0]['name']
    out = next(o['name'] for o in emb_meta['schema']['outputs'] if o['shape'] == [1,200])
    embedder = es.TensorflowPredictMusiCNN(graphFilename=emb_pb,
                                           input=inp, output=out,
                                           patchSize=187, patchHopSize=93, batchSize=0)
    embeddings = embedder(audio16)
    n, d = embeddings.shape
    if n < required_patches:
        pad = np.zeros((required_patches - n, d))
        embeddings = np.vstack([embeddings, pad])
    elif n > required_patches:
        embeddings = embeddings[:required_patches]
    # DEAM head
    with open(head_json) as f:
        head_meta = json.load(f)
    labels = head_meta['classes']
    inp2 = head_meta['schema']['inputs'][0]['name']
    out2 = head_meta['schema']['outputs'][0]['name']
    head = es.TensorflowPredict2D(graphFilename=head_pb,
                                  input=inp2, output=out2, batchSize=0)
    act = head(embeddings)
    arr = np.asarray(act)
    if arr.ndim == 1:
        arr = arr[None, :]

    mean_act = arr.mean(axis=0)  # shape = (2,)
    return float(mean_act[labels.index('arousal')]), float(mean_act[labels.index('valence')])

def get_tempo(y: np.ndarray, sr: int = 22050) -> int:
    """Estimate tempo (BPM) with librosa."""
    return int(librosa.beat.tempo(y=y, sr=sr, max_tempo=200)[0])

# Example Usage
if __name__ == "__main__":
    wav = "/root/autodl-tmp/CLAP/data/classical_multi_dataset/25/4.wav"
    clip_prob = load_panns(wav)
    y = load_librosa(wav)
    audio16 = load_essentia(wav)
    features = {
        "genre": get_genre(clip_prob),
        "mood": get_mood(clip_prob).split()[0].lower(),
        "danceability": round(get_danceability(audio16), 2),
        "energy": round(get_energy(y), 2),
        "acousticness": round(get_acousticness(y), 2),
    }
    aro, val = get_arousal_valence(audio16)
    features.update({
        "valence": round(val, 2),
        "arousal": round(aro, 2),
        "tempo": get_tempo(y)
    })
    print(features)

