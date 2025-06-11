import os
import madmom
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import librosa
import numpy as np
import soundfile as sf

def segment_by_downbeat(
    audio_path,
    bars_per_segment=4,
    beats_per_bar=4,
    fade_ms=50,
    project_dir="F:\\classical_multi_dataset",
    min_d=12.0,    # 最小时长
    max_d=18.0     # 最大时长
):
    """
    使用 madmom 下拍检测将音乐分割，且每段时长限制在 [min_d, max_d] 秒。
    输出到 project_dir/<basename>/0.wav, 1.wav, ...
    """
    # 1) 准备输出目录
    _, audio_filename = os.path.split(audio_path)
    audio_basename, _ = os.path.splitext(audio_filename)
    output_dir = os.path.join(project_dir, audio_basename)
    os.makedirs(output_dir, exist_ok=True)

    # 2) 下拍检测
    proc = RNNDownBeatProcessor(fps=100)
    act  = proc(audio_path)
    dbn  = DBNDownBeatTrackingProcessor(beats_per_bar=beats_per_bar, fps=100)
    downbeats = dbn(act)[:, 0]  # 只要时间戳

    # 3) 如果没检测到任何下拍，整段输出
    y, sr = librosa.load(audio_path, sr=None)
    total_dur = len(y) / sr
    if len(downbeats) == 0:
        sf.write(os.path.join(output_dir, "0.wav"), y, sr)
        print("[WARNING] 未检测到下拍，直接输出全曲")
        return

    # 4) 原始小节起点列表（每 bars_per_segment 小节合并）
    raw_bounds = []
    for i in range(0, len(downbeats), bars_per_segment):
        raw_bounds.append(downbeats[i])
    # 确保包含曲末
    if raw_bounds[-1] < total_dur:
        raw_bounds.append(total_dur)

    # 5) 在 raw_bounds 上做贪心合并/拆分，保证每段在 [min_d, max_d]
    bounds = [raw_bounds[0]]
    i = 1
    while i < len(raw_bounds):
        # 找到最小 j 使段长 >= min_d
        j = i
        while j < len(raw_bounds) and raw_bounds[j] - bounds[-1] < min_d:
            j += 1
        # 如果到末尾，还不够 min_d，就强制收尾
        if j >= len(raw_bounds):
            bounds.append(raw_bounds[-1])
            break
        length = raw_bounds[j] - bounds[-1]
        if length > max_d:
            # 超长则在 max_d 处插入人工切点
            cut = bounds[-1] + max_d
            bounds.append(cut)
            # 把新边界插回 raw_bounds，以后可当作真实边界
            raw_bounds.insert(j, cut)
            # 下一轮从这个新切点开始
            i = j
        else:
            # 合法长度，直接取这个边界
            bounds.append(raw_bounds[j])
            i = j + 1

    # 6) 导出每段
    for idx in range(len(bounds)-1):
        s, e = bounds[idx], bounds[idx+1]
        start_sample = int(s * sr)
        end_sample   = int(e * sr)
        segment      = y[start_sample:end_sample]
        out_path     = os.path.join(output_dir, f"{idx}.wav")
        sf.write(out_path, segment, sr)
        print(f"[Segment {idx}] {s:.2f}s → {e:.2f}s  ({e-s:.2f}s) -> {out_path}")

    print(f"\n完成：共 {len(bounds)-1} 段，保存在 {output_dir}")

if __name__ == "__main__":
    segment_by_downbeat(
        r"F:\musicdataset\Baroque\4.mp3",
        bars_per_segment=4,
        beats_per_bar=4,
        fade_ms=50,
        project_dir=r"F:\musicdataset\Baroque",
        min_d=12.0,
        max_d=18.0
    )
