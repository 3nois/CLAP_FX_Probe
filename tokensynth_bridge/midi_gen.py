"""9차 F-1 — 소스별 MIDI 생성 (수정: 소스당 3변형).

기존 파이프라인은 전 소스에 같은 MIDI 프레이즈(input_midi.mid)를 썼다 — 베이스와
플루트가 같은 음역을 연주하면 당연히 어색하고, 지속음 위주라 잔향이 드러날 여백도
없었다. 이 모듈은 소스의 NSynth pitch를 중심으로 짧은 프레이즈(3~4음, 각
0.4~0.8초)를 만들고 마지막 음 뒤에 1.5초 이상의 무음을 남긴다.

★ 소스당 서로 다른 형태의 프레이즈 3개(variant)를 만든다 — "MIDI가 결과를
좌우하는가"를 분리하기 위해서다. 세 변형은 같은 4개 pitch(중심±5반음, 소스마다
한 번만 추첨)를 재사용하되 순서(윤곽)만 다르게 배열한다 — pitch 집합 자체가
아니라 멜로디 윤곽만 변수로 남겨 깨끗하게 비교할 수 있다.

    variant 0  상행(낮은 음 -> 높은 음, 오름차순)
    variant 1  하행(내림차순, variant 0의 역순)
    variant 2  도약(지그재그 — 낮음/높음을 번갈아, 정순의 양끝에서 안쪽으로)
"""
from pathlib import Path

import numpy as np
import pretty_midi

NOTE_DUR = 0.6
NOTE_GAP = 0.05
N_NOTES = 4
PITCH_RANGE_SEMITONES = 5
MIN_TRAILING_SILENCE = 1.5
N_VARIANTS = 3


def _order_pitches(sorted_offsets, variant: int):
    if variant == 0:
        return list(sorted_offsets)
    elif variant == 1:
        return list(reversed(sorted_offsets))
    elif variant == 2:
        # 지그재그: 정렬된 리스트의 양끝에서 안쪽으로 번갈아 뽑는다 (낮음,높음,낮음,높음 ...)
        lo, hi = 0, len(sorted_offsets) - 1
        zigzag = []
        take_low = True
        while lo <= hi:
            if take_low:
                zigzag.append(sorted_offsets[lo]); lo += 1
            else:
                zigzag.append(sorted_offsets[hi]); hi -= 1
            take_low = not take_low
        return zigzag
    else:
        raise ValueError(f"variant는 0/1/2만 지원: {variant}")


def _write_phrase(pitches, velocity, note_dur, gap, out_path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    for p in pitches:
        note = pretty_midi.Note(velocity=int(velocity), pitch=int(p), start=t, end=t + note_dur)
        inst.notes.append(note)
        t += note_dur + gap
    last_note_end = t - gap
    trailing_silence = 5.0 - last_note_end
    pm.instruments.append(inst)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    return last_note_end, trailing_silence


def generate_midi_variants_for_source(center_pitch: int, velocity: int, seed: int, out_dir: Path, tag: str,
                                       n_notes: int = N_NOTES, note_dur: float = NOTE_DUR, gap: float = NOTE_GAP):
    """소스당 3변형(상행/하행/지그재그)을 생성해 저장한다. 4개 pitch는 소스마다 한 번만
    추첨해 세 변형이 공유한다 — 윤곽만 다르게 하기 위함.

    반환: {variant_idx: {"pitches":..., "path":..., "trailing_silence":..., "shape": "ascending"|...}}
    """
    rng = np.random.RandomState(seed)
    offsets = list(range(-PITCH_RANGE_SEMITONES, PITCH_RANGE_SEMITONES + 1))
    chosen = sorted(rng.choice(offsets, size=min(n_notes, len(offsets)), replace=False).tolist())
    pitch_set = [int(np.clip(center_pitch + o, 0, 127)) for o in chosen]

    total_note_span = n_notes * note_dur + (n_notes - 1) * gap
    assert total_note_span + MIN_TRAILING_SILENCE <= 5.0, (
        f"note_dur/gap 설정이 너무 길어 5.0초 안에 {MIN_TRAILING_SILENCE}초 여백을 못 남깁니다 "
        f"(음표 구간 {total_note_span:.2f}초)"
    )

    shape_names = {0: "ascending", 1: "descending", 2: "zigzag"}
    result = {}
    for variant in range(N_VARIANTS):
        ordered = _order_pitches(pitch_set, variant)
        out_path = Path(out_dir) / f"{tag}_v{variant}.mid"
        last_note_end, trailing_silence = _write_phrase(ordered, velocity, note_dur, gap, out_path)
        result[variant] = {
            "pitches": ordered, "shape": shape_names[variant], "path": str(out_path),
            "last_note_end": last_note_end, "trailing_silence": trailing_silence,
            "n_notes": n_notes, "note_dur": note_dur, "gap": gap,
        }
    return result
