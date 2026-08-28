# -*- coding: utf-8 -*-
"""Phase 8-1 — 다중 시드 평균(A) · 변위 증폭(B) · 온도 하향(C) 개입 격자.

out/prereg/11_phase8.md 확정 설계. highshelf_gain 전범위(idx 0=e_dry_true(-15dB),
idx 24=e_wet(+15dB)) 축 하나, 100소스(10패밀리x10, seed=0 층화)에 대해:

  조건 c   e_wet 그대로 주입 — (source,temp,seed)당 1회, alpha 전체에서 재사용
  조건 d   e_inject(alpha) = e_wet + alpha*(e_dry_true-e_wet) 주입 — alpha마다 1회

격자: alpha in {1,2,3,5} x temperature in {1.0,0.7} x seed in {42,43,44,45}.
seed=42는 Phase 6과 동일 — (alpha=1, temp=1.0, seed=42) 슬라이스가 재현 확인이다.

temperature는 TokenSynth.synthesize()에 인자가 없어 tokensynth.utils.sample을
런타임 몽키패치로 노출한다(벤더 코드 미수정, 이미 존재하는 매개변수 노출만).

재사용(그대로 호출): inject.py(synthesize_from_embedding), midi_gen.py(MIDI 1변형).
소스 인터리브 + work-item 단위 체크포인트(JSONL, --resume 지원) — Phase 6과 동일 패턴.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import torch
import audiofile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tokensynth_bridge"))
from importlib import import_module

dr = import_module("11_phase2_doseresponse")
render_mod = import_module("11_phase2_render")

from inject import synthesize_from_embedding
from midi_gen import generate_midi_variants_for_source

import tokensynth.utils as ts_utils

_ORIG_SAMPLE = ts_utils.sample
_CURRENT_TEMPERATURE = [1.0]


def _patched_sample(logits, top_p=None, top_k=None, midi_vocab_size=None, audio_vocab_size=None, temperature=1.0):
    return _ORIG_SAMPLE(logits, top_p=top_p, top_k=top_k, midi_vocab_size=midi_vocab_size,
                         audio_vocab_size=audio_vocab_size, temperature=_CURRENT_TEMPERATURE[0])


ts_utils.sample = _patched_sample  # 몽키패치 — out/prereg/11_phase8.md §0.4

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = dr.RESULTS_DIR
OUT_AUDIO_DIR = REPO_ROOT / "out" / "audio" / "phase8"
NEW_MIDI_DIR = REPO_ROOT / "tokensynth_bridge" / "generated_midi"
CHECKPOINT_PATH = RESULTS_DIR / "11_phase8_checkpoint.jsonl"
PROGRESS_LOG_PATH = RESULTS_DIR / "11_phase8_progress.log"

TOP_K = 10
TS_ENCODE_SR = 16000
AXIS_NAME = "highshelf_gain"
IDX_A, IDX_B = 0, 24  # e_dry_true(-15dB), e_wet(+15dB) — Phase 6과 동일 정의
ALPHAS = [1, 2, 3, 5]
TEMPS = [1.0, 0.7]
SEEDS = [42, 43, 44, 45]
N_PER_FAMILY = 10
SOURCE_SELECT_SEED = 0
SILENCE_PEAK_THRESHOLD = 1e-4
CLIP_THRESHOLD = 0.99
CLIP_FRAC_THRESHOLD = 0.01


def cos_np(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bootstrap_ci_by_source(values, src_ids, seed, n_boot=2000):
    values = np.asarray(values)
    sources = np.unique(src_ids)
    if len(sources) < 2:
        return float(values.mean()) if len(values) else float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    src_to_rows = {s: np.where(src_ids == s)[0] for s in sources}
    means = []
    for _ in range(n_boot):
        boot = rng.choice(sources, size=len(sources), replace=True)
        rows = np.concatenate([src_to_rows[s] for s in boot])
        means.append(values[rows].mean())
    means = np.array(means)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def embed_ts_from_16k(clap_wrapper, device, y16k):
    y48 = librosa.resample(y16k, orig_sr=16000, target_sr=48000).astype(np.float32)
    tensor = torch.tensor(y48.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        return clap_wrapper.clap.get_audio_embedding_from_data(tensor, use_tensor=True)


def all_sources_1200():
    with open(RESULTS_DIR / "11_phase2_sources.json", encoding="utf-8") as f:
        base_sources = json.load(f)["sources"]
    with open(RESULTS_DIR / "11_phase2_sources_ext.json", encoding="utf-8") as f:
        ext_sources = json.load(f)["sources"]
    return sorted(base_sources + ext_sources, key=lambda s: s["src_id"])


def select_100_sources(sources):
    by_family = {}
    for s in sources:
        by_family.setdefault(s["family"], []).append(s["src_id"])
    rng = np.random.RandomState(SOURCE_SELECT_SEED)
    selected = []
    for fam in sorted(by_family.keys()):
        chosen = rng.choice(by_family[fam], size=N_PER_FAMILY, replace=False)
        selected.extend(sorted(int(x) for x in chosen))
    return sorted(selected)


def audio_flags(audio_np):
    peak = float(np.abs(audio_np).max())
    silent = peak < SILENCE_PEAK_THRESHOLD
    clip_frac = float((np.abs(audio_np) >= CLIP_THRESHOLD).mean())
    clipped = clip_frac >= CLIP_FRAC_THRESHOLD
    has_nan = bool(np.isnan(audio_np).any()) or bool(np.isinf(audio_np).any())
    return silent, clipped, clip_frac, has_nan


def load_checkpoint():
    done = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["type"], row["src_pos"], row["temp"], row["seed"], row.get("alpha"))
                done[key] = row
    return done


def log_progress(msg):
    print(msg)
    with open(PROGRESS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 8-1 — A(다중시드)/B(변위증폭)/C(온도) 개입 격자")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-sources", type=int, default=None, help="스모크 테스트용")
    args = parser.parse_args()

    sources = all_sources_1200()
    family_by_pos = {s["src_id"]: s["family"] for s in sources}
    filename_by_pos = {s["src_id"]: s["filename"] for s in sources}

    selected_pos = select_100_sources(sources)
    if args.max_sources is not None:
        selected_pos = selected_pos[: args.max_sources]

    emb, theta_raw, src_id = dr.load_concat(AXIS_NAME)
    assert np.array_equal(src_id, np.arange(1200))
    e_dry_true_all = emb[:, IDX_A, :]
    e_wet_all = emb[:, IDX_B, :]

    total_c = len(selected_pos) * len(TEMPS) * len(SEEDS)
    total_d = len(selected_pos) * len(ALPHAS) * len(TEMPS) * len(SEEDS)
    total_gen = total_c + total_d
    print(f"소스 {len(selected_pos)}개, 조건c {total_c}건 + 조건d {total_d}건 = 총 {total_gen}건")

    done_items = load_checkpoint() if args.resume else {}
    if done_items:
        log_progress(f"체크포인트에서 재개: 완료 {len(done_items)}건")
    elif CHECKPOINT_PATH.exists():
        raise SystemExit(f"체크포인트가 이미 존재({CHECKPOINT_PATH}) — --resume으로 이어가거나 지울 것.")

    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    NEW_MIDI_DIR.mkdir(parents=True, exist_ok=True)

    log_progress("CLAP/TokenSynth/DAC 로딩 중...")
    from tokensynth import TokenSynth, CLAP, DACDecoder
    device = torch.device(args.device)
    synth = TokenSynth.from_pretrained(aug=True, device=device)
    clap = CLAP(device=device)
    decoder = DACDecoder(device=device)
    log_progress("모델 로딩 완료.")

    midi_cache = {}

    def get_midi_path(pos):
        if pos not in midi_cache:
            fname = filename_by_pos[pos]
            _, pitch, velocity = render_mod.parse_nsynth_filename(Path(fname))
            variants = generate_midi_variants_for_source(pitch, velocity, seed=pitch, out_dir=NEW_MIDI_DIR,
                                                           tag=f"p8_pos{pos}")
            midi_cache[pos] = variants[0]["path"]
        return midi_cache[pos]

    n_done = len(done_items)
    t_start = time.time()
    n_done_at_start = n_done
    last_pct = -1
    ckpt_f = open(CHECKPOINT_PATH, "a", encoding="utf-8")

    def write_row(row):
        nonlocal n_done, last_pct
        ckpt_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        ckpt_f.flush()
        key = (row["type"], row["src_pos"], row["temp"], row["seed"], row.get("alpha"))
        done_items[key] = row
        n_done += 1
        pct = int(100 * n_done / total_gen)
        if pct != last_pct:
            last_pct = pct
            elapsed = time.time() - t_start
            rate = (n_done - n_done_at_start) / elapsed if elapsed > 0 else 0
            remaining = (total_gen - n_done) / rate if rate > 0 else float("inf")
            log_progress(f"[{pct:3d}%] {n_done}/{total_gen}  {row['type']} pos={row['src_pos']} "
                         f"temp={row['temp']} seed={row['seed']} alpha={row.get('alpha')} "
                         f"경과={elapsed/3600:.2f}h 잔여={remaining/3600:.2f}h 처리율={rate:.4f}gen/s")

    for pos in selected_pos:
        e_dry_true = e_dry_true_all[pos]
        e_wet = e_wet_all[pos]
        midi_path = get_midi_path(pos)
        family = family_by_pos[pos]

        for temp in TEMPS:
            for seed in SEEDS:
                c_key = ("c", pos, temp, seed, None)
                if c_key not in done_items:
                    _CURRENT_TEMPERATURE[0] = temp
                    tok_c = synthesize_from_embedding(synth, e_wet, midi_path, seed=seed, normalize="none", top_k=TOP_K)
                    with torch.no_grad():
                        audio_c = decoder.decode(tok_c).cpu().numpy()
                    tag = f"highshelf_full_{pos}_t{temp}_s{seed}_c"
                    audiofile.write(str(OUT_AUDIO_DIR / f"{tag}.wav"), audio_c, TS_ENCODE_SR)
                    e_regen_c = embed_ts_from_16k(clap, device, audio_c).cpu().numpy().reshape(-1)
                    silent, clipped, clip_frac, has_nan = audio_flags(audio_c)
                    recon_c = cos_np(e_regen_c, e_wet) if not (silent or has_nan) else None
                    write_row({"type": "c", "src_pos": int(pos), "family": family, "temp": temp, "seed": seed,
                               "alpha": None, "recon_fidelity": recon_c, "silent": silent, "clipped": clipped,
                               "clip_frac": clip_frac, "has_nan": has_nan,
                               "e_regen": None if (silent or has_nan) else e_regen_c.tolist()})

                for alpha in ALPHAS:
                    d_key = ("d", pos, temp, seed, alpha)
                    if d_key in done_items:
                        continue
                    e_inject = e_wet + alpha * (e_dry_true - e_wet)
                    _CURRENT_TEMPERATURE[0] = temp
                    tok_d = synthesize_from_embedding(synth, e_inject, midi_path, seed=seed, normalize="none", top_k=TOP_K)
                    with torch.no_grad():
                        audio_d = decoder.decode(tok_d).cpu().numpy()
                    tag = f"highshelf_full_{pos}_t{temp}_s{seed}_a{alpha}_d"
                    audiofile.write(str(OUT_AUDIO_DIR / f"{tag}.wav"), audio_d, TS_ENCODE_SR)
                    e_regen_d = embed_ts_from_16k(clap, device, audio_d).cpu().numpy().reshape(-1)
                    silent, clipped, clip_frac, has_nan = audio_flags(audio_d)
                    recon_d = cos_np(e_regen_d, e_dry_true) if not (silent or has_nan) else None

                    c_row = done_items[("c", pos, temp, seed, None)]
                    if c_row.get("e_regen") is not None and not (silent or has_nan):
                        e_regen_c = np.array(c_row["e_regen"])
                        v_generated = e_regen_d - e_regen_c
                        v_original = e_dry_true - e_wet
                        da = cos_np(v_generated, v_original)
                    else:
                        da = None

                    write_row({"type": "d", "src_pos": int(pos), "family": family, "temp": temp, "seed": seed,
                               "alpha": alpha, "recon_fidelity": recon_d, "silent": silent, "clipped": clipped,
                               "clip_frac": clip_frac, "has_nan": has_nan, "da": da})

    ckpt_f.close()
    log_progress(f"전체 완료: {n_done}/{total_gen}")


if __name__ == "__main__":
    main()
