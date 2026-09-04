# -*- coding: utf-8 -*-
"""Phase 9 tier-1 청취 도구 생성.

out/audio/phase9_listen/manifest.json 을 읽어 자기완결 HTML 한 장을 만든다.
의존성 없음 — 브라우저에서 파일을 더블클릭하면 바로 돈다(서버 불필요).

설계상 중요한 점 셋:
  1) 블라인드   R0/R1 을 A/B 로 무작위 배정하고 답하기 전까지 정체를 숨긴다.
  2) 음량 정규화 왜곡은 RMS 를 올리므로 음량 차이가 판단을 오염시킨다.
                시행 안 3개 파일의 RMS 를 맞춘 게인을 미리 계산해 심는다.
  3) 기록       시행별 A/B 매핑·응답·소요시간을 남기고 JSON 으로 내보낸다.

출력: out/audio/phase9_listen/listen.html
"""
import json
import math
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LISTEN_DIR = ROOT / "out" / "audio" / "phase9_listen"


def wav_rms(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        n, sw, ch = w.getnframes(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"16-bit PCM 만 지원: {path.name} (sampwidth={sw})")
    total, count = 0, 0
    for i in range(0, len(raw) - 1, 2 * ch):
        v = int.from_bytes(raw[i:i + 2], "little", signed=True)
        total += v * v
        count += 1
    return math.sqrt(total / max(count, 1)) / 32768.0


def build_trials():
    manifest = json.loads((LISTEN_DIR / "manifest.json").read_text(encoding="utf-8"))
    trials = []
    for i, m in enumerate(manifest):
        q, r0, r1 = m["files"]
        rms = {f: wav_rms(LISTEN_DIR / f) for f in (q, r0, r1)}
        floor = min(v for v in rms.values() if v > 1e-9) if any(v > 1e-9 for v in rms.values()) else 1.0
        gain = {f: (min(1.0, floor / v) if v > 1e-9 else 1.0) for f, v in rms.items()}
        trials.append({
            "id": i,
            "axis": m["axis"],
            "level": m["level"],
            "alpha": m["alpha"],
            "query_src": m["query_src"],
            "query": q,
            "R0": {"file": r0, "src": m["R0_top1_src"], "tag": m["R0_top1_tag"]},
            "R1": {"file": r1, "src": m["R1_top1_src"], "tag": m["R1_top1_tag"]},
            "gain": {"query": round(gain[q], 4), "R0": round(gain[r0], 4), "R1": round(gain[r1], 4)},
        })
    return trials


HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>Phase 9 — tier 1 청취</title>
<style>
:root{--bg:#14161a;--fg:#e8eaed;--mut:#9aa0a6;--acc:#6fa8dc;--ok:#7fbf7f;--warn:#e0a458;--line:#2a2e35}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:19px;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:20px}
.bar{height:5px;background:var(--line);border-radius:3px;overflow:hidden;margin-bottom:22px}
.bar>i{display:block;height:100%;background:var(--acc);transition:width .25s}
.meta{color:var(--mut);font-size:12px;letter-spacing:.02em;margin-bottom:14px}
.card{border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}
.card.ref{border-color:#3a4048;background:#191c21}
.lab{font-size:12px;color:var(--mut);margin-bottom:8px}
button{font:inherit;color:var(--fg);background:#222730;border:1px solid var(--line);
 border-radius:8px;padding:9px 14px;cursor:pointer}
button:hover{border-color:var(--acc)}
button.play{min-width:112px}
button.play.on{background:var(--acc);color:#10141a;border-color:var(--acc)}
.q{margin:22px 0 8px;font-weight:600}
.opts{display:flex;gap:8px;flex-wrap:wrap}
.opts button{flex:1;min-width:120px}
.opts button.sel{background:var(--acc);color:#10141a;border-color:var(--acc)}
.nav{display:flex;gap:10px;margin-top:26px;align-items:center}
.nav .sp{flex:1}
.hint{color:var(--mut);font-size:12px;margin-top:18px}
.reveal{margin-top:16px;padding:12px;border:1px dashed var(--line);border-radius:8px;
 font-size:13px;color:var(--mut)}
.reveal b{color:var(--ok)}
#done{display:none}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}
th,td{border-bottom:1px solid var(--line);padding:7px 6px;text-align:left}
th{color:var(--mut);font-weight:600}
.big{font-size:26px;font-weight:700;color:var(--ok)}
</style></head><body><div class="wrap">

<h1>Phase 9 — tier 1 청취</h1>
<div class="sub">블라인드 · A/B 무작위 배정 · 음량 정규화됨</div>
<div class="bar"><i id="prog" style="width:0"></i></div>

<div id="run">
  <div class="meta" id="meta"></div>

  <div class="card ref">
    <div class="lab">기준 — 이펙트가 걸린 원본</div>
    <button class="play" data-k="query">▶︎ 원본 재생</button>
  </div>

  <div class="card">
    <div class="lab">검색 결과</div>
    <div class="opts">
      <button class="play" data-k="A">▶︎ A 재생</button>
      <button class="play" data-k="B">▶︎ B 재생</button>
    </div>
  </div>

  <div class="q">1. 원본과 비교해 <u>이펙트가 덜 걸린</u> 쪽은?</div>
  <div class="opts" id="q1">
    <button data-v="A">A</button><button data-v="tie">차이 없음</button><button data-v="B">B</button>
  </div>

  <div class="q">2. 원본과 <u>음색·악기가 더 비슷한</u> 쪽은?</div>
  <div class="opts" id="q2">
    <button data-v="A">A</button><button data-v="tie">차이 없음</button><button data-v="B">B</button>
  </div>

  <div class="q">3. 심하게 망가진 소리가 있나?</div>
  <div class="opts" id="q3">
    <button data-v="none">없음</button><button data-v="A">A</button>
    <button data-v="B">B</button><button data-v="both">둘 다</button>
  </div>

  <div class="nav">
    <button id="prev">← 이전</button><div class="sp"></div>
    <button id="skip">모르겠음 · 건너뛰기</button>
    <button id="next">다음 →</button>
  </div>

  <div id="rv" class="reveal" style="display:none"></div>

  <div class="hint">
    단축키 &nbsp; <b>1</b> 원본 &nbsp; <b>2</b> A &nbsp; <b>3</b> B &nbsp;·&nbsp;
    <b>Q W E</b> 1번 문항 &nbsp; <b>A S D</b> 2번 문항 &nbsp;·&nbsp;
    <b>Enter</b> 다음 &nbsp; <b>R</b> 정답 보기
  </div>
</div>

<div id="done">
  <h1>완료</h1>
  <div id="summary"></div>
  <div class="nav"><button id="save">결과 JSON 내려받기</button>
    <button id="again">다시 하기</button></div>
</div>

</div>
<script>
const TRIALS = __TRIALS__;
const SEED   = __SEED__;

// 재현 가능한 셔플 (mulberry32)
function rng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);
  t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const rnd = rng(SEED);
function shuffle(a){a=a.slice();for(let i=a.length-1;i>0;i--){const j=Math.floor(rnd()*(i+1));
  [a[i],a[j]]=[a[j],a[i]]}return a}

// 시행 순서와 A/B 배정을 한 번에 확정.
// A/B 는 무작위가 아니라 정확히 반반으로 균형 배정한다 — n=24 에서 무작위로 두면
// 배정이 치우쳐(예: 8/24) 위치 선호(A 를 자주 고르는 습관)와 교락될 수 있다.
const ORDER = shuffle(TRIALS.map((t,i)=>i));
const HALF  = shuffle(TRIALS.map((t,i)=> i < TRIALS.length/2 ? 1 : 0));
const MAP   = HALF.map(h => h ? {A:"R1",B:"R0"} : {A:"R0",B:"R1"});

let idx = 0;
const ans = TRIALS.map(()=>({q1:null,q2:null,q3:null,plays:0,t0:null,ms:0}));
let audio = null, curBtn = null;

const $ = s => document.querySelector(s);
const cur = () => TRIALS[ORDER[idx]];
const curAns = () => ans[ORDER[idx]];

function stop(){ if(audio){audio.pause();audio=null}
  document.querySelectorAll(".play").forEach(b=>b.classList.remove("on")); curBtn=null }

function play(kind){
  const t = cur(), m = MAP[ORDER[idx]];
  const key = kind==="query" ? "query" : m[kind];      // A/B → R0/R1
  const file = kind==="query" ? t.query : t[key].file;
  const gain = kind==="query" ? t.gain.query : t.gain[key];
  const btn = document.querySelector(`.play[data-k="${kind}"]`);
  if(curBtn===btn){ stop(); return }
  stop();
  audio = new Audio(file); audio.volume = gain; audio.play();
  btn.classList.add("on"); curBtn = btn;
  audio.onended = ()=>{ btn.classList.remove("on"); curBtn=null };
  curAns().plays++;
}

function paint(){
  stop();
  const t = cur(), a = curAns();
  if(a.t0===null) a.t0 = performance.now();
  $("#meta").textContent =
    `${idx+1} / ${TRIALS.length}   ·   ${t.axis}   레벨 ${t.level}   α=${t.alpha}   소스 #${t.query_src}`;
  $("#prog").style.width = (idx/TRIALS.length*100)+"%";
  [["#q1","q1"],["#q2","q2"],["#q3","q3"]].forEach(([sel,k])=>{
    document.querySelectorAll(sel+" button").forEach(b=>
      b.classList.toggle("sel", a[k]===b.dataset.v));
  });
  $("#rv").style.display="none";
}

function reveal(){
  const t = cur(), m = MAP[ORDER[idx]];
  const d = k => `${m[k]} · 소스 #${t[m[k]].src} · <b>${t[m[k]].tag}</b>`;
  $("#rv").innerHTML = `A = ${d("A")}<br>B = ${d("B")}`;
  $("#rv").style.display="block";
}

[["#q1","q1"],["#q2","q2"],["#q3","q3"]].forEach(([sel,k])=>{
  document.querySelector(sel).addEventListener("click",e=>{
    const b=e.target.closest("button"); if(!b)return;
    curAns()[k]=b.dataset.v; paint();
  });
});
document.querySelectorAll(".play").forEach(b=>
  b.addEventListener("click",()=>play(b.dataset.k)));

function advance(d){
  const a=curAns(); if(a.t0!==null){a.ms+=performance.now()-a.t0; a.t0=null}
  idx+=d;
  if(idx>=TRIALS.length){ finish(); return }
  if(idx<0) idx=0;
  paint();
}
$("#next").onclick=()=>advance(1);
$("#prev").onclick=()=>advance(-1);
$("#skip").onclick=()=>{const a=curAns();a.q1=a.q1||"skip";a.q2=a.q2||"skip";advance(1)};

document.addEventListener("keydown",e=>{
  if(e.key==="1")play("query"); else if(e.key==="2")play("A"); else if(e.key==="3")play("B");
  else if("qwe".includes(e.key.toLowerCase())){curAns().q1=["A","tie","B"]["qwe".indexOf(e.key.toLowerCase())];paint()}
  else if("asd".includes(e.key.toLowerCase())){curAns().q2=["A","tie","B"]["asd".indexOf(e.key.toLowerCase())];paint()}
  else if(e.key==="Enter")advance(1);
  else if(e.key.toLowerCase()==="r")reveal();
});

function rows(){
  return ORDER.map((ti,pos)=>{
    const t=TRIALS[ti], m=MAP[ti], a=ans[ti];
    const pick = q => a[q]==null?null : (a[q]==="tie"||a[q]==="skip") ? a[q] : m[a[q]];
    return {pos, trial_id:t.id, axis:t.axis, level:t.level, alpha:t.alpha,
      query_src:t.query_src, mapping:m,
      R0_tag:t.R0.tag, R1_tag:t.R1.tag,
      q1_raw:a.q1, q1_arm:pick("q1"), q2_raw:a.q2, q2_arm:pick("q2"),
      q3_raw:a.q3, q3_arm:(a.q3==="A"||a.q3==="B")?m[a.q3]:a.q3,
      plays:a.plays, seconds:+(a.ms/1000).toFixed(1)};
  });
}

function finish(){
  $("#run").style.display="none"; $("#done").style.display="block";
  $("#prog").style.width="100%";
  const R=rows();
  const tally=(k)=>{const c={R0:0,R1:0,tie:0,skip:0,none:0};
    R.forEach(r=>{const v=r[k]; if(v in c)c[v]++}); return c};
  const c1=tally("q1_arm"), c2=tally("q2_arm");
  const dec=c1.R0+c1.R1;
  const rate=dec? (c1.R1/dec*100).toFixed(1) : "—";
  const by={};
  R.forEach(r=>{const k=r.axis+" lvl"+r.level;(by[k]=by[k]||{R0:0,R1:0,tie:0})[r.q1_arm==="R1"?"R1":r.q1_arm==="R0"?"R0":"tie"]++});
  $("#summary").innerHTML =
    `<p>문항 1 — <u>이펙트가 덜 걸린</u> 쪽으로 <b>R1</b>을 고른 비율</p>
     <div class="big">${rate}%</div>
     <p style="color:var(--mut);font-size:13px">
       R1 ${c1.R1} · R0 ${c1.R0} · 차이없음 ${c1.tie} · 건너뜀 ${c1.skip}
       &nbsp;(판정한 ${dec}건 기준, 우연 50%)</p>
     <p style="margin-top:18px">문항 2 — <u>음색이 더 비슷한</u> 쪽<br>
       <span style="color:var(--mut);font-size:13px">R1 ${c2.R1} · R0 ${c2.R0} · 차이없음 ${c2.tie}</span></p>
     <table><tr><th>축 · 레벨</th><th>R1</th><th>R0</th><th>차이없음</th></tr>` +
     Object.entries(by).map(([k,v])=>`<tr><td>${k}</td><td>${v.R1}</td><td>${v.R0}</td><td>${v.tie}</td></tr>`).join("") +
     `</table>
     <p class="hint">표본 24건이므로 이 수치는 <b>정성적 확인용</b>이다.
        정식 판정에는 쓰지 않는다(§8 주장 범위).</p>`;
}

$("#save").onclick=()=>{
  const out={generated:new Date().toISOString(),seed:SEED,n:TRIALS.length,
    note:"Phase 9 tier-1 blind listening. q1=이펙트 적음, q2=음색 유사, q3=파탄",
    rows:rows()};
  const b=new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(b); a.download="phase9_listen_results.json"; a.click();
};
$("#again").onclick=()=>location.reload();

paint();
</script></body></html>
"""


def main():
    trials = build_trials()
    html = HTML.replace("__TRIALS__", json.dumps(trials, ensure_ascii=False)).replace("__SEED__", "20260904")
    out = LISTEN_DIR / "listen.html"
    out.write_text(html, encoding="utf-8")
    dry = sum(1 for t in trials if t["R1"]["tag"] == "dry")
    print(f"생성: {out}")
    print(f"시행 {len(trials)}건 · R1 top1 이 dry 인 경우 {dry}/{len(trials)}")
    print(f"음량 게인 범위: "
          f"{min(min(t['gain'].values()) for t in trials):.3f} ~ 1.000")


if __name__ == "__main__":
    main()
