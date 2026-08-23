# -*- coding: utf-8 -*-
"""Phase 6 (Q5) — 예측된 손잡이 방향이 실제 생성 오디오에 반영되는가.

대상: 11_phase5_q4.md에서 B2가 between 기준선을 넘은 (축,구간) 조합 20개
(5축 x 4구간: 전범위/하위1/3/중위1/3/상위1/3 — 인접구간은 B2 생략이라 대상 아님).
context 조건 없음 (3-1 v4/v5에서 15건 중 14건이 "context 부차적"이라 불필요).

방법(9차 F-4, tokensynth_bridge/phase_f4_full.py와 동일 패턴 — 재구현 아님, 재사용):
  각 축의 구간 (idx_a=시작, idx_b=끝)에서 Phase 2 캐시의 실제 임베딩
  e_dry_true=emb[idx_a], e_wet=emb[idx_b]를 그대로 읽는다(재렌더링/재추출 없음).
  MIDI 1변형으로 TokenSynth를 e_wet(조건c)/e_dry_true(조건d)로 조건화해 각각
  생성 -> 디코드 -> CLAP 재추출.
  v_generated = e_regen_d - e_regen_c, v_original = e_dry_true - e_wet,
  directional_agreement = cos(v_generated, v_original).

재사용(그대로 호출, 재구현 금지): inject.py(synthesize_from_embedding),
midi_gen.py(generate_midi_variants_for_source), phase_f2_filter.py(is_allowed,
11차 Phase6에서 lowshelf/peak=전패밀리 추가됨), phase_f4_full.py의
embed_ts_from_16k/cos_np/bootstrap_ci_by_source 패턴.

실행 설계(사용자 지시, 2026-08-22 — 실측 규모가 예상 4,600건이 아니라 41,280건,
약 54~70시간으로 드러난 뒤 확정):
  1. ★ 소스 인터리브: 바깥 루프=소스, 안쪽 루프=그 소스가 해당하는 조합. 조합
     우선으로 돌면 중단 시점에 일부 조합이 n=0으로 남는다 — 소스 인터리브면
     언제 멈춰도 20개 조합이 거의 같은 n을 갖는다.
  2. 체크포인트: 소스 완료마다 JSONL에 증분 저장, --resume으로 이어서 진행.
     1% 단위 진행 로그(완료 소스 수, 조합별 누적 n, 경과/잔여 시간, 처리율).
  3. n 불균형: 결과표에 축별 full n(600/960/1200)과 균형 서브샘플(600, 고정
     시드) 버전을 나란히 낸다.
  4. 중간 보고: 완료 소스 300/600명 시점에 인터림 리포트 자동 생성. 최고
     성능 조합의 평균 da가 SANITY_THRESHOLD 미만이면(파이프라인 결함 의심)
     자동 중단 + 명시적 오류 메시지.
  5. 처리율(초/생성, 소스/시간)을 매 체크포인트마다 로그에 남긴다(3일 연속
     부하 시 처리량이 떨어질 수 있음 — Phase 3 전례).
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
from phase_f2_filter import is_allowed

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = dr.RESULTS_DIR
OUT_AUDIO_DIR = REPO_ROOT / "out" / "audio" / "phase6"
NEW_MIDI_DIR = REPO_ROOT / "tokensynth_bridge" / "generated_midi"
CHECKPOINT_PATH = RESULTS_DIR / "11_phase6_checkpoint.jsonl"
PROGRESS_LOG_PATH = RESULTS_DIR / "11_phase6_progress.log"

GEN_SEED = 42
TS_ENCODE_SR = 16000
BALANCED_N = 600
BALANCED_SEED = 0
SANITY_THRESHOLD = 0.05  # 최고 성능 조합 평균 da가 이 미만이면 파이프라인 결함 의심 -> 자동 중단
INTERIM_CHECKPOINTS = [300, 600]  # 완료 소스 수 기준

AXIS_TO_EFFECT = {
    "distortion_drive_db": "distortion",
    "reverb_room_size": "reverb",
    "highshelf_gain": "highshelf",
    "lowshelf_gain": "lowshelf",
    "peak_gain": "peak",
}
INTERVALS = [("전범위", 0, 24), ("하위1/3", 0, 8), ("중위1/3", 8, 16), ("상위1/3", 16, 24)]
LABEL_TAG = {"전범위": "full", "하위1/3": "low", "중위1/3": "mid", "상위1/3": "high"}

PREREG_PREDICTION = (
    "구간별 예측: 전범위 > 상위1/3 > 중위1/3 > 하위1/3 순으로 directional_agreement가 "
    "높을 것(5-C의 B2 cos 순서와 동일 순위 유지를 예측). 특히 하위1/3은 임베딩에서도 "
    "가장 약했던 구간이라, 오디오 경로의 추가 잡음(자기회귀 샘플링 등)까지 더해지면 "
    "무작위 수준(cos~0)에 가까워질 위험이 가장 크다고 예측한다."
)


def cos_np(a, b):
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
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


def build_combos():
    combos = []
    for axis_name, effect in AXIS_TO_EFFECT.items():
        for label, idx_a, idx_b in INTERVALS:
            combos.append({"combo_id": f"{axis_name}::{label}", "axis": axis_name, "effect": effect,
                            "interval": label, "idx_a": idx_a, "idx_b": idx_b})
    return combos


def load_lf_energy():
    with open(RESULTS_DIR / "11_source_lf_energy.json", encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v for k, v in d["lf_rms_by_src_id"].items()}


def load_checkpoint():
    done = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                done[(row["src_pos"], row["combo_id"])] = row
    return done


def log_progress(msg):
    print(msg)
    with open(PROGRESS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def write_interim_report(all_rows, n_sources_done, elapsed_sec, tag):
    lines = [f"# Phase 6 인터림 리포트 — 완료 소스 {n_sources_done}명 (경과 {elapsed_sec/3600:.1f}h) — {tag}\n"]
    lines.append(f"**사전 등록 예측**: {PREREG_PREDICTION}\n")
    lines.append("| 축 | 구간 | n | directional_agreement 평균(95%CI) |")
    lines.append("|---|---|---|---|")
    best_mean = -2.0
    for axis_name in AXIS_TO_EFFECT:
        for label, _, _ in INTERVALS:
            sub = [r for r in all_rows if r["axis"] == axis_name and r["interval"] == label]
            if not sub:
                lines.append(f"| {axis_name} | {label} | 0 | — |")
                continue
            vals = np.array([r["da"] for r in sub])
            srcs = np.array([r["src_pos"] for r in sub])
            m, lo, hi = bootstrap_ci_by_source(vals, srcs, seed=0)
            best_mean = max(best_mean, m)
            lines.append(f"| {axis_name} | {label} | {len(sub)} | {m:+.4f} [{lo:+.4f},{hi:+.4f}] |")
    out_path = RESULTS_DIR / f"11_phase6_interim_{tag}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log_progress(f"인터림 리포트 저장: {out_path} (최고 조합 평균 da={best_mean:+.4f})")
    return best_mean


def write_final_report(all_rows):
    lf_energy = load_lf_energy()
    lines = ["# Phase 6 (Q5) — directional_agreement 결과 (2026-08-22)\n"]
    lines.append(f"**사전 등록 예측**: {PREREG_PREDICTION}\n")
    lines.append("## 축별 전체 n(필터 통과분 그대로)\n")
    lines.append("| 축 | 구간 | n | directional_agreement 평균(95%CI) | LF에너지 상관(rho, lowshelf만) |")
    lines.append("|---|---|---|---|---|")
    for axis_name in AXIS_TO_EFFECT:
        for label, _, _ in INTERVALS:
            sub = [r for r in all_rows if r["axis"] == axis_name and r["interval"] == label]
            if not sub:
                continue
            vals = np.array([r["da"] for r in sub])
            srcs = np.array([r["src_pos"] for r in sub])
            m, lo, hi = bootstrap_ci_by_source(vals, srcs, seed=0)
            lf_note = ""
            if axis_name == "lowshelf_gain":
                lf_vals = np.array([lf_energy.get(r["src_pos"]) for r in sub])
                from scipy import stats as _stats
                rho, p = _stats.spearmanr(vals, lf_vals)
                lf_note = f"{rho:+.3f} (p={p:.2e})"
            lines.append(f"| {axis_name} | {label} | {len(sub)} | {m:+.4f} [{lo:+.4f},{hi:+.4f}] | {lf_note} |")

    lines.append(f"\n## 균형 서브샘플 비교 (축마다 n={BALANCED_N}, 시드={BALANCED_SEED})\n")
    lines.append("축 간 표본 구성이 다르므로(필터가 이펙트마다 다른 수의 패밀리를 배제) "
                 f"직접 비교용으로 모든 축을 n={BALANCED_N}으로 균형 서브샘플한 버전을 병기한다.\n")
    lines.append("| 축 | 구간 | n(균형) | directional_agreement 평균(95%CI) |")
    lines.append("|---|---|---|---|")
    rng = np.random.RandomState(BALANCED_SEED)
    for axis_name in AXIS_TO_EFFECT:
        for label, _, _ in INTERVALS:
            sub = [r for r in all_rows if r["axis"] == axis_name and r["interval"] == label]
            if len(sub) < BALANCED_N:
                lines.append(f"| {axis_name} | {label} | {len(sub)}(<{BALANCED_N}, 전체 사용) | — |")
                continue
            idx = rng.choice(len(sub), size=BALANCED_N, replace=False)
            sub_b = [sub[i] for i in idx]
            vals = np.array([r["da"] for r in sub_b])
            srcs = np.array([r["src_pos"] for r in sub_b])
            m, lo, hi = bootstrap_ci_by_source(vals, srcs, seed=0)
            lines.append(f"| {axis_name} | {label} | {BALANCED_N} | {m:+.4f} [{lo:+.4f},{hi:+.4f}] |")

    lines.append("\n**주의**: 필터가 이펙트마다 다른 수의 패밀리를 배제하므로(위 n열 참고) "
                 "전체-n 표는 축 간 직접 비교에 쓰지 말고 균형 서브샘플 표를 참고할 것 "
                 "(예: distortion/reverb는 일부 악기 패밀리가 원천 배제, EQ 3종은 전 패밀리).\n")

    out_path = RESULTS_DIR / "11_phase6_directional_agreement.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log_progress(f"최종 리포트 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 — directional_agreement (실제 오디오 검증, 소스 인터리브 + 체크포인트)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-sources-per-combo", type=int, default=None, help="스모크 테스트용")
    parser.add_argument("--axes", type=str, default=None, help="스모크 테스트용")
    parser.add_argument("--intervals", type=str, default=None, help="스모크 테스트용")
    parser.add_argument("--dry-run-count", action="store_true")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 이어서 진행")
    parser.add_argument("--max-source-count", type=int, default=None, help="이번 실행에서 처리할 소스 수 상한(스모크/부분 실행용)")
    parser.add_argument("--cpu-threads", type=int, default=None,
                         help="torch.set_num_threads() — CPU 점유율 제한용(cpulimit이 멀티스레드 torch에 안 먹혀서 대체)")
    args = parser.parse_args()

    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        log_progress(f"torch 스레드 수를 {args.cpu_threads}로 제한 (CPU 점유율 억제)")

    sources = all_sources_1200()
    family_by_pos = {s["src_id"]: s["family"] for s in sources}
    filename_by_pos = {s["src_id"]: s["filename"] for s in sources}

    combos = build_combos()
    if args.axes:
        wanted = set(args.axes.split(","))
        combos = [c for c in combos if c["axis"] in wanted]
    if args.intervals:
        wanted = set(args.intervals.split(","))
        combos = [c for c in combos if c["interval"] in wanted]

    combo_by_id = {c["combo_id"]: c for c in combos}
    eligible_by_combo = {}
    for c in combos:
        elig = [pos for pos in range(1200) if is_allowed(c["effect"], family_by_pos[pos])]
        if args.max_sources_per_combo is not None:
            elig = elig[: args.max_sources_per_combo]
        eligible_by_combo[c["combo_id"]] = set(elig)

    all_pos = sorted(set().union(*eligible_by_combo.values())) if combos else []
    if args.max_source_count is not None:
        all_pos = all_pos[: args.max_source_count]

    total_work_items = sum(len(eligible_by_combo[cid] & set(all_pos)) for cid in combo_by_id)
    total_gen = total_work_items * 2
    print(f"사전 등록 예측: {PREREG_PREDICTION}\n")
    for cid, c in combo_by_id.items():
        n = len(eligible_by_combo[cid] & set(all_pos))
        print(f"  {c['axis']:20s} {c['interval']:8s} n={n:4d}")
    print(f"\n소스 수(인터리브 대상) = {len(all_pos)}, 조합 수 = {len(combos)}, "
          f"총 work-item = {total_work_items}, 총 생성 수 = {total_gen}")
    if args.dry_run_count:
        return

    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    NEW_MIDI_DIR.mkdir(parents=True, exist_ok=True)

    done_items = load_checkpoint() if args.resume else {}
    if done_items:
        log_progress(f"체크포인트에서 재개: 완료된 work-item {len(done_items)}개")
    elif CHECKPOINT_PATH.exists() and not args.resume:
        raise SystemExit(f"체크포인트 파일이 이미 존재합니다({CHECKPOINT_PATH}) — --resume으로 이어가거나 "
                          f"백업 후 삭제하세요. 실수로 덮어써서 이전 진행분을 잃지 않도록 하는 안전장치입니다.")

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
            variants = generate_midi_variants_for_source(
                pitch, velocity, seed=pitch, out_dir=NEW_MIDI_DIR, tag=f"p6_pos{pos}")
            midi_cache[pos] = variants[0]["path"]
        return midi_cache[pos]

    emb_cache = {}

    def get_emb(axis_name):
        if axis_name not in emb_cache:
            emb, theta_raw, src_id = dr.load_concat(axis_name)
            assert np.array_equal(src_id, np.arange(1200))
            emb_cache[axis_name] = emb
        return emb_cache[axis_name]

    all_rows = list(done_items.values())
    n_gen_done = len(all_rows) * 2
    completed_src_set = set()
    for cid, elig in eligible_by_combo.items():
        pass  # per-source completeness computed below

    def source_fully_done(pos):
        for cid in combo_by_id:
            if pos in eligible_by_combo[cid] and (pos, cid) not in done_items:
                return False
        return True

    n_sources_done = sum(1 for pos in all_pos if source_fully_done(pos))
    interim_fired = {k for k in INTERIM_CHECKPOINTS if k <= n_sources_done}

    t_start = time.time()
    n_gen_done_at_start = n_gen_done  # 재개 시 이전 실행분을 이번 세션 경과시간으로 나누지 않도록 분리
    last_pct_reported = -1
    ckpt_f = open(CHECKPOINT_PATH, "a", encoding="utf-8")

    for src_idx, pos in enumerate(all_pos):
        if source_fully_done(pos):
            continue
        for cid, c in combo_by_id.items():
            if pos not in eligible_by_combo[cid]:
                continue
            if (pos, cid) in done_items:
                continue
            axis_name, label, idx_a, idx_b = c["axis"], c["interval"], c["idx_a"], c["idx_b"]
            emb = get_emb(axis_name)
            e_dry_true = emb[pos, idx_a, :]
            e_wet = emb[pos, idx_b, :]
            midi_path = get_midi_path(pos)
            tag = f"{axis_name}_{LABEL_TAG[label]}_{pos}"

            tok_c = synthesize_from_embedding(synth, e_wet, midi_path, seed=GEN_SEED, normalize="none", top_k=args.top_k)
            with torch.no_grad():
                audio_c = decoder.decode(tok_c).cpu().numpy()
            audiofile.write(str(OUT_AUDIO_DIR / f"{tag}_c.wav"), audio_c, TS_ENCODE_SR)
            e_regen_c = embed_ts_from_16k(clap, device, audio_c).cpu().numpy().reshape(-1)

            tok_d = synthesize_from_embedding(synth, e_dry_true, midi_path, seed=GEN_SEED, normalize="none", top_k=args.top_k)
            with torch.no_grad():
                audio_d = decoder.decode(tok_d).cpu().numpy()
            audiofile.write(str(OUT_AUDIO_DIR / f"{tag}_d.wav"), audio_d, TS_ENCODE_SR)
            e_regen_d = embed_ts_from_16k(clap, device, audio_d).cpu().numpy().reshape(-1)

            v_generated = e_regen_d - e_regen_c
            v_original = e_dry_true - e_wet
            da = cos_np(v_generated, v_original)

            row = {"axis": axis_name, "interval": label, "combo_id": cid, "src_pos": int(pos),
                   "family": family_by_pos[pos], "da": da}
            all_rows.append(row)
            done_items[(pos, cid)] = row
            ckpt_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            n_gen_done += 2

            pct = int(100 * n_gen_done / total_gen)
            if pct != last_pct_reported:
                last_pct_reported = pct
                elapsed = time.time() - t_start
                rate = (n_gen_done - n_gen_done_at_start) / elapsed if elapsed > 0 else 0
                remaining = (total_gen - n_gen_done) / rate if rate > 0 else float("inf")
                log_progress(f"[{pct:3d}%] gen {n_gen_done}/{total_gen}  {tag} da={da:+.4f}  "
                             f"경과={elapsed/3600:.2f}h 잔여={remaining/3600:.2f}h 처리율={rate:.3f}gen/s")

        if source_fully_done(pos):
            n_sources_done += 1
            for k in INTERIM_CHECKPOINTS:
                if k not in interim_fired and n_sources_done >= k:
                    interim_fired.add(k)
                    elapsed = time.time() - t_start
                    best_mean = write_interim_report(all_rows, n_sources_done, elapsed, tag=f"n{k}")
                    if best_mean < SANITY_THRESHOLD:
                        ckpt_f.close()
                        log_progress(f"★★★ 자동 중단: n={k}에서 최고 조합 평균 da={best_mean:+.4f} < "
                                     f"SANITY_THRESHOLD={SANITY_THRESHOLD} — 파이프라인 결함 의심. "
                                     f"11_phase6_interim_n{k}.md를 확인하고 --resume 전에 원인을 진단할 것.")
                        raise SystemExit(1)

    ckpt_f.close()
    log_progress(f"전체 완료: work-item {len(done_items)}/{total_work_items}")
    write_final_report(all_rows)


if __name__ == "__main__":
    main()
