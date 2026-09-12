"""Read-only projections, playback state, and render helpers for the dashboard.

Playback state is UI-local only. The final synthesis is rendered as an independent finale.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from html import escape
from math import ceil
from collections.abc import Mapping, MutableMapping
from typing import Any, Callable

import plotly.graph_objects as go
import streamlit as _streamlit

from src.services.evolution_presentation import load_official_evolution_run
from src.ui.narrative_evolution import narrative_change
from src.ui.scroll_continuity import render_scroll_continuity


_CANDIDATES = (
    ("EC-ADD-01-口味图鉴", "口味图鉴", "#9FE0D1"),
    ("EC-ADD-02-梅见溯源记", "梅见溯源记", "#DFA0A3"),
    ("EC-03-双容量双剧本", "双容量双剧本", "#E6C889"),
)
_CANDIDATE_IDS = frozenset(item[0] for item in _CANDIDATES)
_PLATFORM_AVATARS = {
    "小红书": ("xiaohongshu", "小红"),
    "抖音": ("douyin", "抖"),
    "B站": ("bilibili", "B"),
    "京东": ("jd", "京东"),
    "淘宝": ("taobao", "淘"),
}
_NUMERIC_NODE_COUNT = 6
_FINALE_SCENE_LEVELS = {
    "PRIMARY": "主要",
    "SECONDARY": "次要",
    "EXTENSION": "拓展",
}
_FINALE_SLOGAN_LINES = (
    "梅见，懂中国饭，",
    "也懂你怎么喝。",
)
_FINALE_SHORT_EXPLANATION = (
    "理解中国饭桌上的口味差异，建立看得见的信任，尊重每一桌不同的饮用节奏。"
)


def _display_stage_label(index: int) -> str:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < _NUMERIC_NODE_COUNT:
        raise ValueError("展示阶段索引必须在 0 到 5 之间")
    return "初始状态" if index == 0 else f"阶段 {index}"

LIVE_PAGE_CSS = """
[data-testid='stAppViewContainer']{background:#0D1417;color:#ECE9E3;}
[data-testid='stMainBlockContainer']{max-width:96rem;padding-top:.4rem;padding-bottom:.7rem;}
.stApp h1,.stApp h2,.stApp h3{color:#ECE9E3;}
.stButton > button,[data-testid="stButton"] button{background:#19262B;color:#CDD4D3;border:1px solid #354449;box-shadow:none;}
.stButton > button:hover,[data-testid="stButton"] button:hover{border-color:#9FE0D1;color:#9FE0D1;background:#202F34;}
.stButton > button:disabled,[data-testid="stButton"] button:disabled{color:#758488;background:#151F23;border-color:#263439;}
[data-testid='stSelectbox'] label,[data-testid='stExpander'] summary{color:#CDD4D3!important;}
[data-testid='stSelectbox'] > div > div{background:#19262B;color:#ECE9E3;border-color:#354449;}
.mj-live-dashboard{display:grid;grid-template-columns:auto 1fr;align-items:end;gap:.05rem 1rem;margin:0 0 .3rem;padding:.42rem .7rem;border-left:3px solid #9FE0D1;background:linear-gradient(90deg,rgba(32,47,52,.88),rgba(13,20,23,.15));}
.mj-live-kicker{color:#9FE0D1;font-size:.62rem;font-weight:700;letter-spacing:.16em;}
.mj-live-heading{grid-row:2;margin:0!important;font-family:STZhongsong,"华文中宋",serif;font-size:clamp(1.4rem,2vw,2rem)!important;font-weight:500;}
.mj-live-subtitle{grid-column:2;grid-row:1/3;align-self:center;margin:0;color:#94A3A7;font-size:.7rem;line-height:1.55;}
.mj-live-summary{display:flex;align-items:center;gap:1rem;padding:.6rem .8rem;background:#151F23;}.mj-live-summary b{color:#9FE0D1;font-size:1.25rem}.mj-live-summary span{color:#ECE9E3}.mj-live-summary small{margin-left:auto;color:#94A3A7;}
.mj-live-phase{display:grid;grid-template-columns:auto auto 1fr;align-items:center;gap:.65rem;margin:.4rem 0;padding:.45rem .7rem;border:1px solid #26363B;background:#151F23;}.mj-live-phase span{color:#9FE0D1;font-size:.65rem;letter-spacing:.12em}.mj-live-phase strong{font-size:.78rem}.mj-live-phase p{margin:0;color:#94A3A7;font-size:.67rem;text-align:right;}
.mj-progress-strip{display:flex;flex-wrap:wrap;gap:.35rem .7rem;margin:.25rem 0 .45rem;}.mj-progress-node{display:flex;align-items:center;flex:0 1 9rem;gap:.3rem;min-width:0;color:#66777B;font-size:.6rem}.mj-progress-node i{display:grid;place-items:center;flex:0 0 1.15rem;height:1.15rem;border:1px solid #354449;border-radius:50%;font-style:normal}.mj-progress-node b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:500}.mj-progress-node.is-visible{color:#CDD4D3}.mj-progress-node.is-current,.mj-progress-node.is-receiving{color:#9FE0D1}.mj-progress-node.is-current i,.mj-progress-node.is-receiving i{border-color:#9FE0D1;box-shadow:0 0 12px rgba(159,224,209,.22)}.mj-progress-node.is-open{flex:1 1 12rem;color:#7E9094}.mj-progress-node.is-open i{border-style:dashed}.mj-progress-node.is-receiving i{animation:mjProgressReceive 1.4s ease-in-out infinite}
.mj-live-left,.mj-live-right{min-height:27.5rem;padding:.65rem;border:1px solid #26363B;background:#111B1F;box-sizing:border-box;}.mj-live-right{display:flex;flex-direction:column;gap:.5rem;}
.mj-comment-idle{display:grid;place-items:center;height:8.5rem;border:1px dashed #354449;color:#758488;font-size:.72rem;background:#0F181B;}
.mj-blind-confirmation{height:auto;min-height:8.5rem;padding:1rem;box-sizing:border-box;border:1px solid rgba(223,160,163,.28);background:#1A2226;}.mj-blind-confirmation span{color:#DFA0A3;font-size:.65rem;letter-spacing:.12em}.mj-blind-confirmation h3{margin:.45rem 0 .25rem;font-size:1rem}.mj-blind-confirmation p{margin:0;color:#94A3A7;font-size:.68rem;}
.mj-narrative-stage{min-height:24rem;padding:1.15rem 1.2rem;border-left:3px solid #DFA0A3;background:#162126;}.mj-narrative-stage>span{color:#DFA0A3;font-size:.62rem;letter-spacing:.1em}.mj-narrative-stage h3{margin:.45rem 0 .9rem;font-family:STZhongsong,"华文中宋",serif;font-size:1.2rem}.mj-narrative-old,.mj-narrative-new,.mj-narrative-static{color:#D7DEDC;font-family:KaiTi,"楷体",serif;font-size:.94rem;line-height:1.9}.mj-narrative-old{max-height:12rem;margin-bottom:.45rem;overflow:hidden;animation:mjNarrativeOldOut .38s ease both}.mj-narrative-new{margin-top:0!important;animation:mjNarrativeNewIn .42s ease .3s both}.mj-narrative-stage.is-static .mj-narrative-old{display:none}.mj-narrative-stage.is-static .mj-narrative-new{margin-top:0!important;animation:none!important}.mj-narrative-added{color:#FFF4F5;text-decoration:underline;text-decoration-color:#DFA0A3;text-decoration-thickness:2px;text-underline-offset:.18em;background:rgba(223,160,163,.08)}.mj-narrative-removed{text-decoration:line-through;color:#8B9896}
.mj-score-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:.45rem;padding:.75rem;background:#151F23}.mj-score-summary article{display:grid;gap:.2rem;padding:.55rem;background:#19262B}.mj-score-summary small{color:#94A3A7;font-size:.6rem}.mj-score-summary strong{color:#ECE9E3;font-size:1rem}
/* Streamlit 会移除 Markdown 内嵌样式，弹幕运动统一放在页面级 CSS。 */
.mj-evidence-ticker{min-height:10.4rem;padding:.45rem 0 .2rem}.mj-evidence-ticker h3{margin:.05rem 0 .35rem!important;font-size:.86rem!important;font-weight:600}.mj-comment-stage{display:grid;gap:.22rem;overflow:hidden;-webkit-mask-image:linear-gradient(90deg,transparent,#000 4%,#000 96%,transparent);mask-image:linear-gradient(90deg,transparent,#000 4%,#000 96%,transparent)}.mj-comment-lane{position:relative;height:2.2rem;overflow:hidden}.mj-comment-bullet{position:absolute;left:88%;display:inline-flex;align-items:center;gap:.5rem;width:max-content;max-width:70rem;animation:mjCommentGlide 20s linear 0s 1 forwards}.mj-comment-bullet.is-item-0{animation-duration:20s;animation-delay:0s}.mj-comment-bullet.is-item-1{animation-duration:19s;animation-delay:.7s}.mj-comment-bullet.is-item-2{animation-duration:22s;animation-delay:1.4s}.mj-comment-bullet.is-item-3{animation-duration:18s;animation-delay:6.4s}.mj-comment-bullet.is-item-4{animation-duration:23s;animation-delay:7.2s}.mj-platform-avatar{display:grid;place-items:center;flex:0 0 1.8rem;height:1.8rem;border-radius:50%;color:#fff;font-size:.56rem;font-weight:800}.mj-platform-avatar.is-xiaohongshu{background:#C3394F}.mj-platform-avatar.is-douyin{background:#111;border:1px solid #25F4EE}.mj-platform-avatar.is-bilibili{background:#007FA8}.mj-platform-avatar.is-jd{background:#B9362D}.mj-platform-avatar.is-taobao{background:#C94E22}.mj-comment-copy{display:flex;align-items:center;gap:.5rem;padding:.42rem .7rem;border:1px solid rgba(159,224,209,.28);border-radius:1.2rem;background:#202F34;color:#CDD4D3;font-size:.66rem;white-space:nowrap}.mj-comment-copy strong{color:#9FE0D1}.mj-comment-complete{margin-top:.25rem;color:#9FE0D1;font-size:.62rem}.mj-evidence-ticker details{margin-top:.12rem;color:#94A3A7;font-size:.62rem}.mj-evidence-ticker__record{padding:.2rem 0}.mj-evidence-ticker__record p{display:inline;margin-left:.45rem;color:#C9D2D0}.mj-evidence-ticker__platform{color:#9FE0D1}.mj-comment-stage:hover .mj-comment-bullet,.mj-comment-stage:focus-within .mj-comment-bullet{animation-play-state:paused}
@keyframes mjCommentGlide{from{transform:translateX(0)}to{transform:translateX(calc(-100vw - 100%))}}
@keyframes mjProgressReceive{0%,100%{opacity:.55}50%{opacity:1}}
@keyframes mjNarrativeOldOut{from{opacity:1;max-height:8rem;margin-bottom:.45rem}to{opacity:0;visibility:hidden;max-height:0;margin-bottom:0}}@keyframes mjNarrativeNewIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.mj-full-detail{margin:.75rem 0 0;padding:.85rem 1rem;border:1px solid #26363B;background:#111B1F}.mj-full-detail>header{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:.55rem}.mj-full-detail>header span{color:#9FE0D1;font-size:.62rem;letter-spacing:.12em}.mj-full-detail>header small{color:#758488;font-size:.6rem}.mj-full-dimensions{display:grid;border-top:1px solid #26363B}.mj-dimension-row{display:grid;grid-template-columns:7.4rem minmax(8rem,.8fr) 2.5rem minmax(12rem,1.6fr);align-items:center;gap:.65rem;padding:.5rem 0;border-bottom:1px solid #223137}.mj-dimension-row span{color:#CDD4D3;font-size:.65rem}.mj-dimension-row strong{color:#ECE9E3;font-size:.82rem;font-variant-numeric:tabular-nums}.mj-dimension-row i{display:block;height:3px;background:#2B3A3F}.mj-dimension-row i b{display:block;height:100%;background:#9FE0D1}.mj-dimension-row small{color:#A9B7B3;font-size:.61rem;line-height:1.5}.mj-full-impact{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin-top:.7rem}.mj-full-impact section{padding:.2rem 0 .2rem .8rem;border-left:1px solid #2A3A40;background:transparent}.mj-full-impact h4{margin:0 0 .45rem;color:#DFA0A3;font-size:.68rem}.mj-full-impact ul{display:grid;gap:.35rem;margin:0;padding:0;list-style:none}.mj-full-impact li{padding:.25rem 0;color:#CDD4D3;background:transparent;font-size:.62rem;line-height:1.5}.mj-full-impact li strong,.mj-full-impact li span{display:block}.mj-full-impact li span{margin-top:.18rem;color:#94A3A7}
@media(max-width:1100px){.mj-live-subtitle{display:none}.mj-live-dashboard{grid-template-columns:1fr}.mj-live-left,.mj-live-right{min-height:auto}.mj-dimension-row{grid-template-columns:6.6rem minmax(7rem,1fr) 2.5rem}.mj-dimension-row small{grid-column:1/-1}.mj-full-impact{grid-template-columns:1fr}.mj-full-impact ul{grid-template-columns:1fr}}
.mj-kpi-band{display:flex;align-items:stretch;margin:.4rem 0 .5rem;border-top:1px solid #26363B;border-bottom:1px solid #26363B}
.mj-kpi-band article{flex:1;display:grid;gap:.18rem;padding:.62rem .95rem;border-right:1px solid #26363B}
.mj-kpi-band article:last-child{border-right:0}
.mj-kpi-band small{color:#7E9094;font-size:.62rem;letter-spacing:.16em}
.mj-kpi-band strong{color:#ECE9E3;font-family:Georgia,serif;font-size:1.8rem;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.08;animation:mjKpiFlash .14s ease-out both}
.mj-kpi-band .mj-kpi-delta.is-up{color:#9FE0D1}
.mj-kpi-band .mj-kpi-delta.is-down{color:#DFA0A3}
.mj-kpi-band .mj-kpi-node{font-family:"Microsoft YaHei","PingFang SC",sans-serif;font-size:1rem;color:#9FE0D1;letter-spacing:.04em;animation:none}
@keyframes mjKpiFlash{from{opacity:.2}to{opacity:1}}
.mj-candidate-nav{display:flex;align-items:baseline;justify-content:space-between;margin:.55rem 0 .25rem}.mj-candidate-nav span{color:#9FE0D1;font-size:.62rem;letter-spacing:.12em}.mj-candidate-nav small{color:#7E9094;font-size:.6rem}
/* cockpit chrome（自美化层迁入：页面级作用域，避免全页 :has 重算） */
.mj-system-header{background:#111B1F;border-bottom-color:#26363B;box-shadow:none}
.mj-system-header__brand span{color:#9FE0D1}
.mj-system-header__brand strong{color:#ECE9E3}
.mj-system-status b{color:#CDD4D3}
.mj-system-status small{color:#7E8F94}
.mj-system-status article{border-left-color:#26363B}
.mj-system-status i{background:#9FE0D1;box-shadow:0 0 0 4px rgba(159,224,209,.12)}
[data-testid="stExpander"]>details{background:rgba(17,27,31,.45);border-color:#26363B}
[data-testid="stExpander"] summary{color:#CDD4D3}
[data-testid="stButton"]>button{height:2.2rem;font-size:.72rem;border-radius:.3rem}
[data-testid="stButton"]>button[data-testid="baseButton-secondary"]{background:rgba(25,38,43,.55);border-color:#26363B;color:#9FB3AE}
[data-testid="stButton"]>button[data-testid="baseButton-secondary"]:hover{border-color:#9FE0D1;color:#9FE0D1;background:#202F34}
[data-testid="stButton"]>button p{font-size:inherit}
.st-key-evolution_candidate_EC-ADD-01-口味图鉴 button,.st-key-evolution_candidate_EC-ADD-02-梅见溯源记 button,.st-key-evolution_candidate_EC-03-双容量双剧本 button{height:2.65rem!important;justify-content:flex-start;padding:0 .85rem;border-left:3px solid var(--mj-candidate-line)!important;background:#121D21!important;color:#C8D1CF!important;text-align:left}
.st-key-evolution_candidate_EC-ADD-01-口味图鉴 button{--mj-candidate-line:#9FE0D1}.st-key-evolution_candidate_EC-ADD-02-梅见溯源记 button{--mj-candidate-line:#DFA0A3}.st-key-evolution_candidate_EC-03-双容量双剧本 button{--mj-candidate-line:#E6C889}
[class*="st-key-evolution_candidate_"] button:disabled{opacity:1!important;border-color:var(--mj-candidate-line)!important;background:#1B2B30!important;color:#F1F4F3!important}
[class*="st-key-evolution_candidate_"] button:hover{border-color:var(--mj-candidate-line)!important;color:#FFF!important;background:#1B2B30!important}
[class*="st-key-evolution_previous"] button,[class*="st-key-evolution_next"] button,[class*="st-key-evolution_replay"] button{height:2.25rem!important}
.js-plotly-plot .legendtext{fill:#9FB3AE!important}
.js-plotly-plot .g-xtitle text,.js-plotly-plot .g-ytitle text{fill:#8FA09B!important}
:is(.mj-live-subtitle,.mj-live-phase p,.mj-score-summary small,.mj-dimension-row small,.mj-full-detail>header small,.mj-full-impact li span,.mj-comment-idle,.mj-blind-confirmation p,.mj-progress-node){color:#A9B7B3}
.mj-progress-node:not(.is-visible):not(.is-current){color:#7E9094}
.mj-live-phase{border-left:2px solid #9FE0D1;border-top:0;border-right:0;border-bottom:1px solid #26363B;padding-left:.8rem}
.mj-live-left,.mj-live-right{border-color:#22313a}
/* de-box：看板次级容器发丝线化，主图表区独享抬升 */
.mj-score-summary{background:transparent}
.mj-score-summary article{background:transparent;border-right:1px solid #22313a}
.mj-score-summary article:last-child{border-right:0}
.mj-full-impact section{background:transparent;border:0;border-left:1px solid #22313a}
.mj-full-impact li{background:rgba(25,38,43,.4)}
.mj-full-detail,.mj-live-left,.mj-live-right{border-color:#1D2B30}
.mj-comment-idle{height:3.1rem;min-height:3.1rem}.mj-live-panel{padding:.52rem .75rem;border-radius:0}
section.main{position:relative;z-index:1}
[data-testid="stPlotlyChart"]{position:relative;overflow:hidden}
[data-testid="stPlotlyChart"]::after{content:"";position:absolute;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,rgba(255,255,255,.026) 0 1px,transparent 1px 3px)}
.mj-live-ambience{position:fixed;inset:-6%;z-index:0;pointer-events:none;background-image:radial-gradient(38rem 24rem at 20% 24%,rgba(159,224,209,.05),transparent 70%),radial-gradient(44rem 28rem at 80% 76%,rgba(223,160,163,.045),transparent 70%),linear-gradient(rgba(159,224,209,.026) 1px,transparent 1px),linear-gradient(90deg,rgba(159,224,209,.026) 1px,transparent 1px);background-size:auto,auto,2.6rem 2.6rem,2.6rem 2.6rem}
@media(max-width:900px){[data-testid="stHorizontalBlock"]:has(.mj-live-panel){flex-wrap:wrap!important}[data-testid="stHorizontalBlock"]:has(.mj-live-panel)>[data-testid="column"]{flex:1 1 100%!important;width:100%!important}.mj-narrative-stage{min-height:auto}.mj-kpi-band{display:grid;grid-template-columns:1fr 1fr}.mj-kpi-band article:nth-child(2){border-right:0}}
@media(prefers-reduced-motion:reduce){.mj-comment-bullet{position:static;animation:none!important}.mj-comment-lane{height:auto}.mj-comment-stage{gap:.45rem}}
""".strip()

LATEST_SEGMENT_ANIMATION_CSS = """
.js-plotly-plot .scatterlayer > .trace:nth-of-type(n+4) path.js-line{stroke-dasharray:1000;stroke-dashoffset:1000;animation:mjLineGrow 2.6s cubic-bezier(.2,.75,.25,1) .12s forwards}
.js-plotly-plot .scatterlayer > .trace:nth-of-type(n+4) .point:last-child{transform-box:fill-box;transform-origin:center;animation:mjPointArrive .28s ease 2.45s both}
@keyframes mjLineGrow{to{stroke-dashoffset:0}}
@keyframes mjPointArrive{from{opacity:0;transform:scale(.35)}60%{opacity:1;transform:scale(1.35)}to{opacity:1;transform:scale(1)}}
@media(prefers-reduced-motion:reduce){.js-plotly-plot .scatterlayer path.js-line,.js-plotly-plot .scatterlayer .point{animation:none!important}}
""".strip()

FINALE_PAGE_CSS = """
[data-testid='stAppViewContainer']{background:#3E2C33;color:#FFF8EF;}
[data-testid='stMainBlockContainer'],[data-testid='stAppViewBlockContainer']{max-width:100%;padding:0!important;}
.mj-workspace-nav,[class*="st-key-workspace-nav-"]{display:none!important;}
[data-testid="element-container"]:has(.mj-workspace-nav) + [data-testid="stHorizontalBlock"]{display:none!important;}
.stButton > button,[data-testid="stButton"] button{background:#624751;color:#FFF8EF;border:1px solid #8B6976;box-shadow:none;}
.stButton > button:hover,[data-testid="stButton"] button:hover{background:#725360;color:#FFF8EF;border-color:#C58FA2;}
[data-testid='stExpander'] summary{color:#D5C295!important;}
.mj-finale-shell{min-height:82vh;display:flex;align-items:center;text-align:left;color:#FFF8EF;background:radial-gradient(circle at 24% 35%,rgba(139,105,118,.35),transparent 34%),linear-gradient(145deg,#3E2C33,#513942 58%,#3E2C33);overflow:hidden;}
.mj-finale-curtain{position:relative;width:min(52rem,82vw);margin-left:8vw;padding:5vh 0 4vh;}.mj-finale-curtain::before{content:"见";position:absolute;right:-18vw;top:-9vh;color:rgba(255,248,239,.025);font-family:STZhongsong,"华文中宋",serif;font-size:34rem;line-height:1;}
.mj-finale-kicker{display:block;margin-bottom:1rem;color:#D5C295;font-size:.68rem;font-weight:700;letter-spacing:.22em;}
.mj-finale-slogan{display:grid;gap:.1rem;width:fit-content;font-family:STZhongsong,"华文中宋",serif;font-size:clamp(3rem,5.4vw,5.4rem);font-weight:700;line-height:1.13;letter-spacing:.02em;}.mj-finale-line{display:block;white-space:nowrap}.mj-hanging-mark{display:inline-block;width:.35em;color:#D5C295;transform:translateX(.08em)}
.mj-finale-narrative{max-width:44rem;margin:1.35rem 0 0;color:#FFF8EF;font-size:1rem;line-height:1.8;}
.mj-finale-cta{display:block;margin-top:1.4rem;color:#CFA1B1;font-size:.72rem;letter-spacing:.12em;}
.mj-finale-act-two{margin-top:1.25rem;border-top:1px solid rgba(213,194,149,.42);animation:mjFinaleRise .5s ease both}.mj-finale-pillars{display:grid;grid-template-columns:repeat(3,1fr);}.mj-finale-pillar{min-height:7rem;padding:.85rem 1rem;border-right:1px solid rgba(213,194,149,.32);}.mj-finale-pillar:last-child{border-right:0}.mj-finale-pillar>span{color:#D5C295;font-size:.62rem}.mj-finale-pillar h3{margin:.32rem 0;color:#FFF8EF!important;font-family:STZhongsong,"华文中宋",serif;font-size:1.05rem}.mj-finale-pillar p{margin:0;color:#F3E7E2;font-size:.72rem;line-height:1.65}
.mj-finale-guardrails,.mj-finale-drawer-scenes{display:grid;grid-template-columns:repeat(2,1fr);gap:.8rem;color:#FFF8EF}.mj-finale-drawer-scenes{grid-template-columns:repeat(3,1fr);}.mj-finale-drawer-scenes article,.mj-finale-guardrails section{padding:.85rem;border:1px solid #8B6976;background:#513942}.mj-finale-drawer-scenes span{color:#D5C295;font-size:.65rem}.mj-finale-drawer-scenes h3,.mj-finale-guardrails h3{color:#FFF8EF!important}.mj-finale-drawer-scenes p,.mj-finale-guardrails li{color:#F3E7E2;font-size:.75rem;line-height:1.65}
@keyframes mjFinaleRise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
/* finale chrome（自美化层迁入：页面级作用域） */
.mj-system-header{background:#241B20;border-bottom-color:#57414B}
.mj-system-header__brand span{color:#D5C295}
.mj-system-header__brand strong{color:#FFF3EA}
.mj-system-status b{color:#F0E2DC}
.mj-system-status small{color:#A08C93}
.mj-system-status article{border-left-color:#57414B}
.mj-system-status i{background:#D5C295;box-shadow:0 0 0 4px rgba(213,194,149,.14)}
[data-testid="stExpander"]>details{background:rgba(62,44,51,.55);border-color:#8B6976}
[data-testid="stExpander"] summary{color:#D5C295!important}
.mj-finale-shell{position:relative}
.mj-finale-shell::before{content:"";position:absolute;top:15%;left:50%;width:0;height:1px;background:linear-gradient(90deg,transparent,#D5C295,transparent);box-shadow:0 0 14px rgba(213,194,149,.45);opacity:0;animation:mjPolishGoldLine 1.1s var(--mj-ease-curtain,cubic-bezier(.65,0,.35,1)) .45s forwards;pointer-events:none}
.mj-finale-shell::after{content:"";position:absolute;inset:0;background:radial-gradient(58% 46% at 50% 60%,rgba(213,194,149,.15),transparent 72%);animation:mjPolishGlowUp 2.6s ease-out .7s both;pointer-events:none}
.mj-finale-curtain::before{transform-origin:center;animation:mjPolishBigBreath 9s ease-in-out 2.2s infinite}
.mj-finale-line-one{animation:mjPolishWipeIn .85s cubic-bezier(.65,0,.35,1) .55s both}
.mj-finale-line-two{animation:mjPolishWipeIn .85s cubic-bezier(.65,0,.35,1) .85s both}
.mj-hanging-mark{width:.24em;transform:translateX(0);margin-right:-.16em}
[data-testid="stButton"]>button{height:2.4rem;border-radius:999px;background:rgba(255,248,239,.06);border:1px solid rgba(213,194,149,.45);color:#E9DCC0;font-size:.78rem;letter-spacing:.14em;transition:background .2s ease-out,border-color .2s ease-out}
[data-testid="stButton"]>button:hover{background:rgba(213,194,149,.16);border-color:#D5C295;color:#FFF8EF}
[data-testid="stButton"]>button p{letter-spacing:inherit}
[data-testid="stButton"]>button[data-testid="baseButton-primary"]{max-width:15rem;margin-left:auto;display:block}
@keyframes mjPolishGoldLine{from{left:50%;width:0;opacity:0}30%{opacity:1}to{left:8%;width:84%;opacity:1}}
@keyframes mjPolishGlowUp{0%{opacity:0}45%{opacity:1}100%{opacity:.4}}
@keyframes mjPolishWipeIn{from{clip-path:inset(0 100% 0 0);opacity:.25;transform:translateY(16px)}to{clip-path:inset(0 0 0 0);opacity:1;transform:translateY(0)}}
@keyframes mjPolishBigBreath{0%,100%{transform:scale(1)}50%{transform:scale(1.015)}}
@media(max-width:900px){.mj-finale-curtain{width:86vw;margin:0 auto}.mj-finale-slogan{font-size:clamp(2.35rem,9vw,4rem)}.mj-finale-pillars,.mj-finale-drawer-scenes,.mj-finale-guardrails{grid-template-columns:1fr}.mj-finale-pillar{border-right:0;border-bottom:1px solid rgba(213,194,149,.32)}}
""".strip()


@dataclass(frozen=True)
class FinaleSceneView:
    """Immutable presentation projection for one final synthesis scene."""

    level: str
    scene_id: str
    title: str
    body: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinaleView:
    """Immutable presentation projection for the final synthesis."""

    slogan_lines: tuple[str, str]
    slogan_html: str
    scenes: tuple[FinaleSceneView, ...]
    remaining_risks: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    source_checkpoint_id: str
    core_narrative: str
    pillars: tuple[Mapping[str, Any], ...] = ()
    decision_brief: str = ""


def build_finale_view(final_synthesis: Any) -> FinaleView:
    """Project frozen final-synthesis data into an immutable finale view."""

    raw_scenes = getattr(final_synthesis, "scenes", None)
    if not isinstance(raw_scenes, (tuple, list)):
        raise ValueError("final synthesis must provide scenes with a scene tier")

    scenes: list[FinaleSceneView] = []
    for index, scene in enumerate(raw_scenes):
        if isinstance(scene, Mapping):
            get_value = scene.get
        else:
            get_value = lambda name: getattr(scene, name, None)

        tier = get_value("tier")
        level = _FINALE_SCENE_LEVELS.get(tier)
        if level is None:
            raise ValueError(f"scene tier is missing or unsupported at index {index}: {tier!r}")

        scene_id = get_value("scene_id")
        title = get_value("label")
        body = get_value("action")
        if not all(isinstance(value, str) and value for value in (scene_id, title, body)):
            raise ValueError(f"final scene {index} is missing scene_id, label, or action")

        evidence_refs = get_value("evidence_refs")
        if evidence_refs is None:
            refs = ()
        elif isinstance(evidence_refs, (tuple, list)) and all(
            isinstance(ref, str) and ref for ref in evidence_refs
        ):
            refs = tuple(evidence_refs)
        else:
            raise ValueError(f"final scene {index} evidence_refs is invalid")

        scenes.append(
            FinaleSceneView(
                level=level,
                scene_id=scene_id,
                title=title,
                body=body,
                evidence_refs=refs,
            )
        )

    source_checkpoint_id = getattr(final_synthesis, "source_checkpoint_id", None)
    core_narrative = getattr(final_synthesis, "core_narrative", None)
    if not isinstance(source_checkpoint_id, str) or not source_checkpoint_id:
        raise ValueError("final synthesis source_checkpoint_id is invalid")
    if not isinstance(core_narrative, str) or not core_narrative:
        raise ValueError("final synthesis core_narrative is invalid")

    def string_tuple(field_name: str) -> tuple[str, ...]:
        value = getattr(final_synthesis, field_name, None)
        if not isinstance(value, (tuple, list)) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(f"final synthesis {field_name} is invalid")
        return tuple(value)

    pillars = getattr(final_synthesis, "pillars", ())
    if not isinstance(pillars, (tuple, list)) or not all(
        isinstance(pillar, Mapping) for pillar in pillars
    ):
        raise ValueError("final synthesis pillars is invalid")

    decision_brief = getattr(final_synthesis, "decision_brief", "")
    if not isinstance(decision_brief, str):
        raise ValueError("final synthesis decision_brief is invalid")

    return FinaleView(
        slogan_lines=_FINALE_SLOGAN_LINES,
        slogan_html=_build_finale_slogan_html(_FINALE_SLOGAN_LINES),
        scenes=tuple(scenes),
        remaining_risks=string_tuple("remaining_risks"),
        forbidden_claims=string_tuple("forbidden_claims"),
        source_checkpoint_id=source_checkpoint_id,
        core_narrative=core_narrative,
        pillars=tuple(pillars),
        decision_brief=decision_brief,
    )


def _build_finale_slogan_html(lines: tuple[str, str]) -> str:
    line_markup = []
    for class_name, line in zip(
        ("mj-finale-line-one", "mj-finale-line-two"),
        lines,
        strict=True,
    ):
        line_markup.append(
            f'<span class="mj-finale-line {class_name}">'
            f'{escape(line[:-1])}'
            f'<span class="mj-hanging-mark">{escape(line[-1])}</span>'
            "</span>"
        )
    return (
        f'<div class="mj-finale-slogan" aria-label="{escape("".join(lines), quote=True)}">'
        + "".join(line_markup)
        + "</div>"
    )


def build_finale_html(finale: FinaleView, *, act: int = 1) -> str:
    """Build one of the two rose finale acts; audit boundaries stay in the drawer."""

    if act not in {1, 2}:
        raise ValueError("终幕 act 只能是 1 或 2")

    pillar_markup = "".join(
        '<article class="mj-finale-pillar">'
        f"<span>0{index}</span><h3>{escape(str(pillar.get('name', '叙事支柱')))}</h3>"
        f"<p>{escape(str(pillar.get('role', '')))}</p></article>"
        for index, pillar in enumerate(finale.pillars, start=1)
    )
    act_two = (
        '<div class="mj-finale-act-two">'
        f'<div class="mj-finale-pillars">{pillar_markup}</div></div>'
        if act == 2
        else '<span class="mj-finale-cta">展开完整叙事</span>'
    )
    return (
        f'<section class="mj-finale-shell" data-act="{act}"><div class="mj-finale-curtain">'
        '<span class="mj-finale-kicker">梅见最终品牌方向</span>'
        f"{finale.slogan_html}"
        f'<p class="mj-finale-narrative">{escape(_FINALE_SHORT_EXPLANATION)}</p>'
        f"{act_two}"
        '</div></section>'
    )


class PlaybackAction(str, Enum):
    """Actions understood by the pure playback reducer."""

    PLAY = "PLAY"
    PAUSE = "PAUSE"
    PREVIOUS = "PREVIOUS"
    NEXT_CHANGE = "NEXT_CHANGE"
    REPLAY = "REPLAY"
    SELECT_CHECKPOINT = "SELECT_CHECKPOINT"
    SELECT_CANDIDATE = "SELECT_CANDIDATE"


class PlaybackPhase(str, Enum):
    """One visible change inside a numeric checkpoint."""

    COMMENTS = "comments"
    SCORE = "score"
    NARRATIVE = "narrative"


_PHASE_ORDER = (
    PlaybackPhase.COMMENTS,
    PlaybackPhase.SCORE,
    PlaybackPhase.NARRATIVE,
)


@dataclass(frozen=True)
class PlaybackState:
    """Immutable UI-only playback state; it never represents a seventh score node."""

    checkpoint_index: int
    phase: PlaybackPhase
    playing: bool
    finale: bool

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_index, int) or isinstance(self.checkpoint_index, bool):
            raise ValueError("checkpoint_index 必须是整数")
        if not 0 <= self.checkpoint_index < _NUMERIC_NODE_COUNT:
            raise ValueError("checkpoint_index 必须在 0 到 5 之间")
        if not isinstance(self.phase, PlaybackPhase):
            raise ValueError("phase 必须是 PlaybackPhase")
        if not isinstance(self.playing, bool) or not isinstance(self.finale, bool):
            raise ValueError("playing 和 finale 必须是布尔值")


def reduce_playback(
    state: PlaybackState,
    action: PlaybackAction,
    value: int | str | None = None,
    *,
    checkpoint_index: int | None = None,
    candidate_id: str | None = None,
) -> PlaybackState:
    """Reduce one UI action without mutating state or frozen artifacts."""

    if not isinstance(state, PlaybackState):
        raise ValueError("state 必须是 PlaybackState")
    if not isinstance(action, PlaybackAction):
        raise ValueError("action 必须是 PlaybackAction")

    if action is PlaybackAction.PLAY:
        return PlaybackState(state.checkpoint_index, state.phase, True, state.finale)
    if action is PlaybackAction.PAUSE:
        return PlaybackState(state.checkpoint_index, state.phase, False, state.finale)
    if action is PlaybackAction.REPLAY:
        return PlaybackState(0, PlaybackPhase.SCORE, False, False)
    if action is PlaybackAction.PREVIOUS:
        return _previous_change(state)
    if action is PlaybackAction.NEXT_CHANGE:
        return _next_change(state)
    if action is PlaybackAction.SELECT_CANDIDATE:
        target_candidate_id = candidate_id if candidate_id is not None else value
        if target_candidate_id is not None and (
            not isinstance(target_candidate_id, str) or not target_candidate_id
        ):
            raise ValueError("candidate_id 必须是非空字符串")
        return PlaybackState(state.checkpoint_index, state.phase, False, state.finale)
    if action is PlaybackAction.SELECT_CHECKPOINT:
        target = checkpoint_index if checkpoint_index is not None else value
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError("选择检查点必须提供整数 checkpoint_index")
        return PlaybackState(target, PlaybackPhase.SCORE, False, False)
    raise ValueError(f"未知播放动作: {action}")


def _next_change(state: PlaybackState) -> PlaybackState:
    if state.finale:
        return PlaybackState(state.checkpoint_index, state.phase, False, True)
    if state.checkpoint_index == 0:
        return PlaybackState(1, PlaybackPhase.COMMENTS, False, False)
    phase_index = _PHASE_ORDER.index(state.phase)
    if phase_index < len(_PHASE_ORDER) - 1:
        return PlaybackState(
            state.checkpoint_index,
            _PHASE_ORDER[phase_index + 1],
            False,
            False,
        )
    if state.checkpoint_index < _NUMERIC_NODE_COUNT - 1:
        return PlaybackState(
            state.checkpoint_index + 1,
            PlaybackPhase.COMMENTS,
            False,
            False,
        )
    return PlaybackState(
        _NUMERIC_NODE_COUNT - 1, PlaybackPhase.NARRATIVE, False, True
    )


def _previous_change(state: PlaybackState) -> PlaybackState:
    if state.finale:
        return PlaybackState(
            _NUMERIC_NODE_COUNT - 1, PlaybackPhase.NARRATIVE, False, False
        )
    if state.checkpoint_index == 0:
        return PlaybackState(0, PlaybackPhase.SCORE, False, False)
    if state.phase is PlaybackPhase.NARRATIVE:
        return PlaybackState(state.checkpoint_index, PlaybackPhase.SCORE, False, False)
    if state.phase is PlaybackPhase.SCORE:
        return PlaybackState(state.checkpoint_index, PlaybackPhase.COMMENTS, False, False)
    previous_index = state.checkpoint_index - 1
    previous_phase = PlaybackPhase.SCORE if previous_index == 0 else PlaybackPhase.NARRATIVE
    return PlaybackState(previous_index, previous_phase, False, False)


@dataclass(frozen=True)
class AttributionProjection:
    """Recorded causes for one candidate at one checkpoint, without inference."""

    checkpoint_index: int
    checkpoint_id: str
    candidate_id: str
    patches: tuple[dict[str, Any], ...]
    risks: tuple[str, ...]
    scenes: tuple[dict[str, Any], ...]
    decision_brief: str


_DIMENSION_LABELS = {
    "evidence_strength": "证据充分度",
    "emotional_tension": "情绪冲突张力",
    "brand_fit_and_exclusivity": "品牌适配与独占性",
    "competitor_difference": "竞品差异度",
    "scene_conversion": "场景转化能力",
}

_INITIAL_RATIONALE_TRANSLATIONS = {
    ("EC-ADD-01-口味图鉴", "evidence_strength"): (
        "Direct user questions about flavors and detailed positive/negative flavor comments are highly relevant; counter-evidence about quality skepticism is also included, making this the most evidence-rich candidate.",
        "用户对不同口味的直接提问，以及详细的正负口感反馈都与议题高度相关；同时纳入了对品质的质疑性反证，因此这是当前证据最充分的候选。",
    ),
    ("EC-ADD-01-口味图鉴", "emotional_tension"): (
        "Choice overload, worry about wasting money, and contradictory taste reviews create solid emotional tension around choosing the right bottle.",
        "选择太多、担心花冤枉钱，以及彼此矛盾的口味评价，共同形成了“如何选对一瓶”的明确情绪张力。",
    ),
    ("EC-ADD-01-口味图鉴", "brand_fit_and_exclusivity"): (
        "The pain point is tied directly to Meijian's multiple SKUs and users' first-purchase decisions; an official guide could be distinctive, but it is not built and competitors may offer similar flavor navigation.",
        "这一痛点直接来自梅见多款产品与用户首次购买时的选择困难；官方图鉴可能形成特色，但目前尚未制作，竞品也可能提供类似的口味导航。",
    ),
    ("EC-ADD-01-口味图鉴", "competitor_difference"): (
        "Avoids claiming flavor variety as a difference and focuses on SKU/process mapping, but whether competitors already offer an equivalent official guide remains unverified.",
        "没有把口味丰富本身包装成差异，而是聚焦产品与工艺的对应关系；但竞品是否已有同类官方指南，仍未核实。",
    ),
    ("EC-ADD-01-口味图鉴", "scene_conversion"): (
        "Concrete shelf-scanning and party-tasting moments can directly guide purchase and trial; conversion depends on unbuilt QR and per-SKU verified data.",
        "货架前扫码和聚会试饮都能直接引导购买与尝试；但转化仍依赖尚未建成的二维码入口，以及逐款核实后的产品数据。",
    ),
    ("EC-ADD-02-梅见溯源记", "evidence_strength"): (
        "Direct distrust comments and some positive craft references establish a real concern, but the core promise of batch-level traceability is not supported by verifiable production records.",
        "用户直接表达的不信任，加上少量对工艺的正面描述，证明问题真实存在；但“按批次可追溯”的核心承诺，尚无可核验的生产记录支撑。",
    ),
    ("EC-ADD-02-梅见溯源记", "emotional_tension"): (
        "The 'is it blended?' distrust and worry about safety/health create high emotional stakes and the strongest tension in the set.",
        "“是不是勾兑的”这类不信任，以及对安全与健康的担忧，形成了很强的情绪压力，是本组候选中张力最强的一项。",
    ),
    ("EC-ADD-02-梅见溯源记", "brand_fit_and_exclusivity"): (
        "Directly targets a core Meijian suspicion and leverages wine-cellar/craft imagery that could feel distinctive; exclusivity is limited because competitors can also use craft-transparency narratives and the traceability is unverified.",
        "直接回应梅见面临的核心质疑，并借助酒窖与酿造工艺画面形成品牌感；但竞品同样可以讲工艺透明，且追溯能力尚未核实，因此专属度受限。",
    ),
    ("EC-ADD-02-梅见溯源记", "competitor_difference"): (
        "Traceability is a plausible proof-point, but similar craft-transparency claims exist among competitors and the candidate explicitly acknowledges the differentiation is not yet established.",
        "追溯可以成为可信证明点，但竞品中也存在类似的工艺透明主张；候选本身也明确承认差异化尚未建立。",
    ),
    ("EC-ADD-02-梅见溯源记", "scene_conversion"): (
        "A documentary and scan-to-verify route could be compelling, but scenes are gated on verified batch records, approvals, and event logistics, delaying actual user engagement.",
        "纪录片与扫码查验的路径很有吸引力，但落地依赖已核实的批次记录、审批和活动执行，因此真实用户触达仍会延后。",
    ),
    ("EC-03-双容量双剧本", "evidence_strength"): (
        "Users explicitly mention both 150ml and 330ml uses, solo relaxation, and small-group drinking; evidence is concrete though actual ongoing availability of both capacities is not verified.",
        "用户明确提到 150 毫升和 330 毫升两种容量，以及独饮放松和小聚使用；证据具体，但两种容量是否持续在售尚未核实。",
    ),
    ("EC-03-双容量双剧本", "emotional_tension"): (
        "Real tension between wanting private micro-drinking and avoiding social judgement or pressure, but expressed in a lighter, everyday tone rather than high-stakes conflict.",
        "想独自小酌，又想避开他人评价与社交压力，构成了真实矛盾；但整体更偏轻松日常，不是高强度冲突。",
    ),
    ("EC-03-双容量双剧本", "brand_fit_and_exclusivity"): (
        "Both capacities and usage contexts appear in Meijian user feedback, making brand fit strong; exclusivity is limited by the acknowledged RIO ownership of solo-drinking occasions and unverified product availability.",
        "两种容量与使用情境都出现在梅见用户反馈中，品牌契合度较强；但独饮场景已被锐澳占据，且产品供应未核实，因此专属度有限。",
    ),
    ("EC-03-双容量双剧本", "competitor_difference"): (
        "Honestly avoids claiming solo drinking as a difference; the dual-capacity short-video concept is a plausible differentiation action but is conditional on actual SKU supply and is easily copied.",
        "没有把独饮强说成品牌差异；双容量短视频是可行的差异化动作，但依赖产品真实供应，也容易被模仿。",
    ),
    ("EC-03-双容量双剧本", "scene_conversion"): (
        "Strongly filmable solo and gathering scripts can support content conversion, but scenes are framed as conditional proposals and cannot be reliably delivered until capacity availability is confirmed.",
        "独饮与小聚脚本都很适合拍摄并推动内容转化；但当前仍是有条件的执行提案，在确认容量供应前无法稳定落地。",
    ),
}


def _localized_dimension_rationale(
    *, candidate_id: str, checkpoint_index: int, dimension_key: str, rationale: str
) -> str:
    if checkpoint_index != 0:
        return rationale
    localized = _INITIAL_RATIONALE_TRANSLATIONS.get((candidate_id, dimension_key))
    if localized is None:
        raise ValueError(f"初始五维理由缺少中文展示映射：{candidate_id}/{dimension_key}")
    expected_source, translated = localized
    if rationale != expected_source:
        raise ValueError(f"初始五维冻结理由已变化：{candidate_id}/{dimension_key}")
    return translated


def initialize_realtime_session(
    session_state: MutableMapping[str, object],
    run: Any,
) -> PlaybackState:
    """Initialize a stable manual playback state without starting timers."""

    if not isinstance(session_state, MutableMapping):
        raise ValueError("session_state 必须是可变映射")
    selected_ids = getattr(run, "selected_candidate_ids", None)
    if not isinstance(selected_ids, (tuple, list)) or not selected_ids:
        raise ValueError("正式演化 view 缺少 selected_candidate_ids")
    candidate_id = session_state.get("evolution_candidate_id")
    if candidate_id is None:
        candidate_id = selected_ids[0]
    if candidate_id not in selected_ids:
        raise ValueError(f"session candidate 不存在于正式演化 view: {candidate_id}")
    session_state["evolution_candidate_id"] = candidate_id

    if not session_state.get("evolution_manual_initialized", False):
        session_state["evolution_manual_initialized"] = True
        session_state["evolution_checkpoint_index"] = 0
        session_state["evolution_phase"] = PlaybackPhase.SCORE.value
        session_state["evolution_playing"] = False
        session_state["evolution_show_finale"] = False
        session_state["evolution_finale_act"] = 1
        session_state["evolution_finale_auto_reveal"] = False
    else:
        session_state.setdefault("evolution_checkpoint_index", 0)
        session_state.setdefault("evolution_phase", PlaybackPhase.SCORE.value)
        session_state.setdefault("evolution_playing", False)
        session_state.setdefault("evolution_show_finale", False)
        session_state.setdefault("evolution_finale_act", 1)
        session_state.setdefault("evolution_finale_auto_reveal", False)
    session_state["evolution_playing"] = False
    session_state["evolution_finale_auto_reveal"] = False

    return _playback_state_from_session(session_state)


def apply_playback_action(
    session_state: MutableMapping[str, object],
    action: PlaybackAction,
    *,
    checkpoint_index: int | None = None,
    candidate_id: str | None = None,
    on_milestone: Callable[[], None] | None = None,
) -> PlaybackState:
    """Apply a UI-only action by writing only the corresponding session keys."""

    state = _playback_state_from_session(session_state)
    next_state = reduce_playback(
        state,
        action,
        checkpoint_index=checkpoint_index,
        candidate_id=candidate_id,
    )
    session_state["evolution_checkpoint_index"] = next_state.checkpoint_index
    session_state["evolution_phase"] = next_state.phase.value
    session_state["evolution_playing"] = False
    session_state["evolution_show_finale"] = next_state.finale
    if action in {PlaybackAction.REPLAY, PlaybackAction.PREVIOUS} or not next_state.finale:
        session_state["evolution_finale_act"] = 1
        session_state["evolution_finale_auto_reveal"] = False
    elif not state.finale and next_state.finale:
        session_state["evolution_finale_act"] = 1
        session_state["evolution_finale_auto_reveal"] = False
    if action is PlaybackAction.SELECT_CANDIDATE and candidate_id is not None:
        session_state["evolution_candidate_id"] = candidate_id
    if on_milestone is not None:
        on_milestone()
    return next_state


def build_checkpoint_detail(
    run: Any,
    *,
    checkpoint_index: int,
    candidate_id: str,
) -> dict[str, Any]:
    """Build the right-rail candidate × checkpoint detail from frozen fields."""

    points = getattr(run, "numeric_points", None)
    if not isinstance(points, (tuple, list)) or not 0 <= checkpoint_index < len(points):
        raise ValueError("checkpoint_index 不在正式 numeric_points 范围内")
    point = points[checkpoint_index]
    candidates = getattr(point, "candidates", None)
    if not isinstance(candidates, (tuple, list)):
        raise ValueError("numeric point 缺少 candidates")
    candidate = next(
        (item for item in candidates if getattr(item, "candidate_id", None) == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"检查点缺少候选 {candidate_id}")

    scores = getattr(candidate, "scores", None)
    if isinstance(scores, dict) and "meijian_fit_and_exclusivity" in scores:
        scores = {
            **scores,
            "brand_fit_and_exclusivity": scores.pop("meijian_fit_and_exclusivity"),
        }
    if not isinstance(scores, dict) or set(scores) != set(_DIMENSION_LABELS):
        raise ValueError(f"候选 {candidate_id} 的五维 scores 不完整")
    dimensions = []
    for key, name in _DIMENSION_LABELS.items():
        score_value = scores[key]
        score = getattr(score_value, "score", None)
        rationale = getattr(score_value, "rationale", None)
        if not isinstance(score, (int, float)) or not isinstance(rationale, str):
            raise ValueError(f"候选 {candidate_id} 的维度 {key} 无效")
        dimensions.append(
            {
                "key": key,
                "name": name,
                "score": score,
                "reason": _localized_dimension_rationale(
                    candidate_id=candidate_id,
                    checkpoint_index=checkpoint_index,
                    dimension_key=key,
                    rationale=rationale,
                ),
            }
        )

    evidence_ids = getattr(point, "new_evidence_ids", None)
    if not isinstance(evidence_ids, (tuple, list)):
        raise ValueError("numeric point 缺少 new_evidence_ids")
    attribution = attribution_projection(run, checkpoint_index, candidate_id)
    text_change = narrative_change(run, candidate_id, checkpoint_index)
    point_label = getattr(point, "label", None)
    if not isinstance(point_label, str) or not point_label:
        raise ValueError("numeric point 缺少有效 label")
    checkpoint_label = _display_stage_label(checkpoint_index)

    return {
        "candidate_id": candidate_id,
        "checkpoint_index": checkpoint_index,
        "checkpoint_id": getattr(point, "checkpoint_id", None),
        "checkpoint_label": checkpoint_label,
        "weighted_score": getattr(candidate, "weighted_score", None),
        "score_change": getattr(candidate, "score_change", None),
        "rank": getattr(candidate, "rank", None),
        "dimensions": dimensions,
        "evidence_ids": list(evidence_ids),
        "risks": list(attribution.risks),
        "scenes": list(attribution.scenes),
        "decision_brief": attribution.decision_brief,
        "patches": list(attribution.patches),
        "narrative_change": text_change,
    }


def build_detail_html(detail: Mapping[str, Any]) -> str:
    """Render one candidate × checkpoint as an editorial right rail."""

    required = {
        "checkpoint_label",
        "weighted_score",
        "score_change",
        "rank",
        "dimensions",
        "risks",
        "scenes",
        "decision_brief",
        "narrative_change",
    }
    missing = sorted(required - set(detail))
    if missing:
        raise ValueError(f"detail 缺少字段: {', '.join(missing)}")
    change = detail["narrative_change"]
    label = getattr(change, "label", None)
    before = getattr(change, "before", None)
    after = getattr(change, "after", None)
    changed = getattr(change, "changed", None)
    diff_html = getattr(change, "diff_html", None)
    if not isinstance(label, str) or not isinstance(before, str) or not isinstance(after, str):
        raise ValueError("narrative_change 无效")
    narrative_transition = build_narrative_transition_html(detail)

    dimensions = detail["dimensions"]
    if not isinstance(dimensions, list) or len(dimensions) != 5:
        raise ValueError("detail dimensions 必须包含五项")
    dimension_markup = "".join(
        "<div class=\"mj-dimension-row\">"
        f"<span>{escape(str(item['name']))}</span>"
        f"<i><b style=\"width:{float(item['score']):.0f}%\"></b></i>"
        f"<strong>{float(item['score']):.0f}</strong>"
        f"<small>{escape(str(item['reason']))}</small>"
        "</div>"
        for item in dimensions
    )
    risks = detail["risks"]
    scenes = detail["scenes"]
    risk_markup = "".join(f"<li>{escape(str(item))}</li>" for item in risks) or "<li>本轮未新增风险</li>"
    scene_markup = "".join(
        f"<li><strong>{escape(str(item.get('label', item.get('scene_id', '场景'))))}</strong>"
        f"<span>{escape(str(item.get('support_summary', '沿用冻结场景判断')))}</span></li>"
        for item in scenes
    ) or "<li><strong>场景结构保持</strong><span>本轮未记录场景变化</span></li>"
    score_change = float(detail["score_change"])
    delta_label = f"{score_change:+.2f}" if score_change else "±0.00"
    return (
        f'<section class="mj-live-detail" data-checkpoint="{escape(str(detail["checkpoint_label"]))}">'
        f"{narrative_transition}"
        '<div class="mj-score-hero">'
        f'<span><small>加权分</small><strong>{float(detail["weighted_score"]):.2f}</strong></span>'
        f'<span><small>本轮变化</small><strong>{delta_label}</strong></span>'
        f'<span><small>当前排名</small><strong>#{int(detail["rank"])}</strong></span></div>'
        f'<div class="mj-dimensions"><h4>五维评分</h4>{dimension_markup}</div>'
        '<div class="mj-impact-grid">'
        f'<section><h4>剩余风险</h4><ul>{risk_markup}</ul></section>'
        f'<section><h4>场景变化</h4><ul>{scene_markup}</ul></section></div>'
        f'<p class="mj-decision-brief">{escape(str(detail["decision_brief"]))}</p>'
        '</section>'
    )


def build_kpi_band_html(detail: Mapping[str, Any] | None) -> str:
    """Top instrument band: borderless big numbers above the cockpit columns."""

    if not isinstance(detail, Mapping):
        return ""
    for key in ("weighted_score", "score_change", "rank", "checkpoint_label"):
        if key not in detail:
            return ""
    change = float(detail["score_change"])
    change_value = f"{change:+.2f}" if change else "±0.00"
    change_class = " is-up" if change > 0 else (" is-down" if change < 0 else "")
    return (
        '<section class="mj-kpi-band">'
        '<article><small>当前加权分</small><strong>'
        f'{float(detail["weighted_score"]):.2f}</strong></article>'
        f'<article><small>本轮变化</small><strong class="mj-kpi-delta{change_class}">'
        f"{change_value}</strong></article>"
        f'<article><small>当前排名</small><strong>#{detail["rank"]}</strong></article>'
        '<article><small>当前节点</small>'
        f'<strong class="mj-kpi-node">{escape(str(detail["checkpoint_label"]))}</strong></article>'
        "</section>"
    )


def build_narrative_transition_html(
    detail: Mapping[str, Any], *, animate: bool = True
) -> str:
    """Keep real before/after layers in the DOM for a visible narrative transition."""

    change = detail.get("narrative_change")
    label = getattr(change, "label", None)
    before = getattr(change, "before", None)
    after = getattr(change, "after", None)
    changed = getattr(change, "changed", None)
    diff_html = getattr(change, "diff_html", None)
    checkpoint_label = detail.get("checkpoint_label")
    if not all(isinstance(value, str) for value in (label, before, after, checkpoint_label)):
        raise ValueError("narrative_change 无效")
    if changed:
        if not isinstance(diff_html, str):
            raise ValueError("发生叙事修改时必须提供真实 diff")
        highlighted_after = diff_html.replace(
            'class="word-diff-add"', 'class="mj-narrative-added"'
        ).replace('class="word-diff-remove"', 'class="mj-narrative-removed"')
        body = (
            f'<div class="mj-narrative-old">{escape(before)}</div>'
            f'<div class="mj-narrative-new">{highlighted_after}</div>'
        )
        stage_class = " is-changing"
    else:
        body = f'<p class="mj-narrative-static">{escape(after)}</p>'
        stage_class = ""
    if not animate:
        stage_class += " is-static"
    return (
        f'<section class="mj-narrative-stage{stage_class}">'
        f'<span>叙事文字舞台 · {escape(label)}</span>'
        f'<h3>{escape(checkpoint_label)}</h3>{body}'
        '<details><summary>查看完整前后对照</summary>'
        f'<div><small>修改前</small><p>{escape(before)}</p></div>'
        f'<div><small>修改后</small><p>{escape(after)}</p></div>'
        "</details></section>"
    )


def build_score_summary_html(detail: Mapping[str, Any]) -> str:
    """Render the compact score/rank block beside the narrative stage."""

    for key in ("weighted_score", "score_change", "rank", "checkpoint_label"):
        if key not in detail:
            raise ValueError(f"detail 缺少字段: {key}")
    score_change = float(detail["score_change"])
    delta_label = f"{score_change:+.2f}" if score_change else "±0.00"
    return (
        '<section class="mj-score-summary">'
        f'<span>当前数据更新 · {escape(str(detail["checkpoint_label"]))}</span>'
        f'<strong>{float(detail["weighted_score"]):.2f}</strong>'
        '<div>'
        f'<p><small>本轮变化</small><b>{delta_label}</b></p>'
        f'<p><small>当前排名</small><b>#{int(detail["rank"])}</b></p>'
        "</div></section>"
    )


def build_dimension_tab_html(detail: Mapping[str, Any]) -> str:
    dimensions = detail.get("dimensions")
    if not isinstance(dimensions, list) or len(dimensions) != 5:
        raise ValueError("detail dimensions 必须包含五项")
    rows = "".join(
        '<div class="mj-dimension-row">'
        f'<span>{escape(str(item["name"]))}</span>'
        f'<i><b style="width:{float(item["score"]):.0f}%"></b></i>'
        f'<strong>{float(item["score"]):.0f}</strong>'
        f'<small>{escape(str(item["reason"]))}</small></div>'
        for item in dimensions
    )
    return f'<section class="mj-detail-tab"><h4>五维变化</h4>{rows}</section>'


def build_evidence_risk_tab_html(detail: Mapping[str, Any]) -> str:
    evidence_ids = detail.get("evidence_ids")
    risks = detail.get("risks")
    if not isinstance(evidence_ids, list) or not isinstance(risks, list):
        raise ValueError("detail 缺少证据与风险")
    risks_markup = "".join(f"<li>{escape(str(item))}</li>" for item in risks)
    if not risks_markup:
        risks_markup = "<li>本轮未新增风险</li>"
    return (
        '<section class="mj-detail-tab"><h4>证据与风险</h4>'
        f'<p class="mj-evidence-count">本轮新增证据 <strong>{len(evidence_ids)}</strong> 条</p>'
        f"<ul>{risks_markup}</ul></section>"
    )


def build_scene_tab_html(detail: Mapping[str, Any]) -> str:
    scenes = detail.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("detail 缺少场景变化")
    items = "".join(
        f'<li><strong>{escape(str(item.get("label", item.get("scene_id", "场景"))))}</strong>'
        f'<span>{escape(str(item.get("support_summary", "沿用冻结场景判断")))}</span></li>'
        for item in scenes
    ) or "<li><strong>场景结构保持</strong><span>本轮未记录场景变化</span></li>"
    return f'<section class="mj-detail-tab"><h4>场景变化</h4><ul>{items}</ul></section>'


def build_full_width_detail_html(detail: Mapping[str, Any]) -> str:
    """Render five dimensions, risks, and scenes as a full-width lower field."""

    dimensions = detail.get("dimensions")
    risks = detail.get("risks")
    scenes = detail.get("scenes")
    if not isinstance(dimensions, list) or len(dimensions) != 5:
        raise ValueError("detail dimensions 必须包含五项")
    if not isinstance(risks, list) or not isinstance(scenes, list):
        raise ValueError("detail 缺少风险或场景")
    dimension_rows = "".join(
        '<div class="mj-dimension-row">'
        f'<span>{escape(str(item["name"]))}</span>'
        f'<i><b style="width:{float(item["score"]):.0f}%"></b></i>'
        f'<strong>{float(item["score"]):.0f}</strong>'
        f'<small>{escape(str(item["reason"]))}</small></div>'
        for item in dimensions
    )
    risk_items = "".join(f"<li>{escape(str(item))}</li>" for item in risks)
    if not risk_items:
        risk_items = "<li>本轮未新增风险</li>"
    scene_items = "".join(
        f'<li><strong>{escape(str(item.get("label", item.get("scene_id", "场景"))))}</strong>'
        f'<span>{escape(str(item.get("support_summary", "沿用冻结场景判断")))}</span></li>'
        for item in scenes
    )
    if not scene_items:
        scene_items = "<li><strong>场景结构保持</strong><span>本轮未记录场景变化</span></li>"
    return (
        '<section class="mj-full-detail"><header><span>五维决策场</span>'
        f'<small>{escape(str(detail.get("checkpoint_label", "当前阶段")))}</small></header>'
        f'<div class="mj-full-dimensions">{dimension_rows}</div>'
        '<div class="mj-full-impact">'
        f'<section><h4>风险观察</h4><ul>{risk_items}</ul></section>'
        f'<section><h4>场景变化</h4><ul>{scene_items}</ul></section>'
        "</div></section>"
    )


def build_attribution_strip_html(detail: Mapping[str, Any]) -> str:
    """Explain what the incoming comments affect before scores are revealed."""

    patches = detail.get("patches")
    risks = detail.get("risks")
    scenes = detail.get("scenes")
    if not all(isinstance(value, list) for value in (patches, risks, scenes)):
        raise ValueError("detail 缺少影响归因字段")
    patch_labels = "、".join(
        escape(str(item.get("field_name", "叙事字段"))) for item in patches
    ) or "本轮文案不调整"
    return (
        '<section class="mj-attribution-strip"><header><span>当前子阶段</span>'
        '<h3>影响归因</h3></header><div>'
        f'<article><small>叙事</small><strong>{patch_labels}</strong></article>'
        f'<article><small>风险</small><strong>{len(risks)} 项持续观察</strong></article>'
        f'<article><small>场景</small><strong>{len(scenes)} 个场景受影响</strong></article>'
        f'</div><p>{escape(str(detail.get("decision_brief", "")))}</p></section>'
    )


def build_dashboard_view(
    run: Any,
    playback_state: PlaybackState,
    *,
    selected_candidate_id: str,
) -> dict[str, Any]:
    """Project the three dashboard layers without owning UI/session state."""

    if not isinstance(playback_state, PlaybackState):
        raise ValueError("playback_state 必须是 PlaybackState")
    points = getattr(run, "numeric_points", None)
    if not isinstance(points, (tuple, list)) or len(points) != _NUMERIC_NODE_COUNT:
        raise ValueError("正式演化 view 必须包含六个 numeric_points")
    selected_ids = getattr(run, "selected_candidate_ids", None)
    if not isinstance(selected_ids, (tuple, list)) or selected_candidate_id not in selected_ids:
        raise ValueError(f"候选不在正式演化 view: {selected_candidate_id}")

    nodes = []
    for index, point in enumerate(points):
        point_id = getattr(point, "checkpoint_id", None)
        point_label = getattr(point, "label", None)
        if not isinstance(point_id, str) or not isinstance(point_label, str):
            raise ValueError("numeric point 缺少 checkpoint_id 或 label")
        nodes.append(
            {
                "index": index,
                "checkpoint_id": point_id,
                "label": _display_stage_label(index),
            }
        )

    detail: dict[str, Any] | None = None
    detail_error: str | None = None
    try:
        detail = build_checkpoint_detail(
            run,
            checkpoint_index=playback_state.checkpoint_index,
            candidate_id=selected_candidate_id,
        )
    except ValueError as exc:
        detail_error = str(exc)

    ticker_html = None
    if (
        playback_state.phase is PlaybackPhase.COMMENTS
        and 2 <= playback_state.checkpoint_index <= 5
    ):
        batches = getattr(run, "delta_batches", None)
        if not isinstance(batches, (tuple, list)) or len(batches) != 4:
            raise ValueError("正式演化 view 必须包含四个 delta_batches")
        ticker_html = build_ticker_html(batches[playback_state.checkpoint_index - 2])

    finale = build_finale_view(getattr(run, "final_synthesis", None))
    visible_checkpoint_index = (
        playback_state.checkpoint_index
        if playback_state.phase in {PlaybackPhase.SCORE, PlaybackPhase.NARRATIVE}
        else max(0, playback_state.checkpoint_index - 1)
    )

    return {
        "active_layer": "finale" if playback_state.finale else "evolution",
        "summary": {
            "title": "AI 推荐 → 真人盲评验证 → 团队确认",
            "candidate_count_before": 5,
            "candidate_count_after": 3,
            "collapsed": True,
        },
        "evolution": {
            "nodes": nodes,
            "checkpoint_index": playback_state.checkpoint_index,
            "checkpoint_id": nodes[playback_state.checkpoint_index]["checkpoint_id"],
            "selected_candidate_id": selected_candidate_id,
            "visible_checkpoint_index": visible_checkpoint_index,
            "phase": playback_state.phase.value,
            "figure": build_score_figure(
                run,
                selected_candidate_id,
                visible_checkpoint_index=visible_checkpoint_index,
                animate_latest_segment=(
                    playback_state.phase is PlaybackPhase.SCORE
                    and visible_checkpoint_index > 0
                ),
            ),
            "ticker_html": ticker_html,
            "detail": detail,
            "detail_error": detail_error,
        },
        "finale": {
            "active": playback_state.finale,
            "checkpoint_id": "checkpoint-04",
            "score_node_count": _NUMERIC_NODE_COUNT,
            "view": finale,
            "title": "最终凝练",
        },
        "playback": playback_state,
    }


def build_progress_strip_html(
    nodes: list[dict[str, Any]], playback_state: PlaybackState
) -> str:
    """Render only occurred checkpoints plus one open-ended continuation."""

    if len(nodes) != _NUMERIC_NODE_COUNT:
        raise ValueError("进度条必须包含六个冻结节点")
    visible_index = (
        playback_state.checkpoint_index
        if playback_state.phase in {PlaybackPhase.SCORE, PlaybackPhase.NARRATIVE}
        else max(0, playback_state.checkpoint_index - 1)
    )
    receiving = (
        playback_state.phase is PlaybackPhase.COMMENTS
        and playback_state.checkpoint_index > visible_index
    )
    last_rendered_index = playback_state.checkpoint_index if receiving else visible_index
    markup = "".join(
        '<span class="mj-progress-node'
        + (" is-visible" if index <= visible_index else "")
        + (" is-current" if index == visible_index and not receiving else "")
        + (" is-receiving" if receiving and index == last_rendered_index else "")
        + f'"><i>{"始" if index == 0 else index}</i><b>{escape(str(node["label"]))}</b></span>'
        for index, node in enumerate(nodes[: last_rendered_index + 1])
    )
    if not receiving and not playback_state.finale and visible_index < len(nodes) - 1:
        markup += (
            '<span class="mj-progress-node is-open"><i>···</i>'
            '<b>下一变化待定 · 可随时收敛</b></span>'
        )
    return f'<div class="mj-progress-strip" aria-label="实时变化阶段进度">{markup}</div>'


def render_realtime_decision_dashboard(
    root: Any,
    *,
    streamlit_module: Any = None,
    _fragment_body: bool = False,
    on_milestone: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Render the read-only dashboard and return its projected view for testing."""

    streamlit_injected = streamlit_module is not None
    st = streamlit_module
    if st is None:
        import streamlit as st  # type: ignore[no-redef]

    run = load_official_evolution_run(root)
    session_state = st.session_state
    state = initialize_realtime_session(session_state, run)
    selected_candidate_id = session_state["evolution_candidate_id"]
    view = build_dashboard_view(
        run,
        state,
        selected_candidate_id=selected_candidate_id,
    )
    if view["active_layer"] == "finale":
        if _fragment_body and hasattr(st, "rerun"):
            st.rerun(scope="app")
            return view
        _render_finale(st, view["finale"]["view"], session_state, on_milestone)
        return view

    if not _fragment_body and hasattr(st, "markdown"):
        st.markdown(
            f"<style>{LIVE_PAGE_CSS}</style>"
            '<div class="mj-live-ambience" aria-hidden="true"></div>'
            "<section class='mj-live-dashboard'>"
            "<div class='mj-live-kicker'>梅见 · 实时决策室</div>"
            "<h1 class='mj-live-heading'>实时叙事演化看板</h1>"
            "<p class='mj-live-subtitle'>正式案例回放。"
            "新增评论先进入影响归因，再更新评分、风险、场景与叙事文字。</p>"
            "</section>",
            unsafe_allow_html=True,
        )

    if not _fragment_body:
        if not streamlit_injected:
            return _render_live_streamlit_fragment(root, on_milestone)
        fragment_renderer = lambda: render_realtime_decision_dashboard(
            root,
            streamlit_module=st,
            _fragment_body=True,
            on_milestone=on_milestone,
        )
        fragment = getattr(st, "fragment", None)
        return fragment(fragment_renderer)() if callable(fragment) else fragment_renderer()

    state = _playback_state_from_session(session_state)
    selected_candidate_id = session_state["evolution_candidate_id"]
    view = build_dashboard_view(
        run,
        state,
        selected_candidate_id=selected_candidate_id,
    )
    if view["active_layer"] == "finale":
        if _fragment_body and hasattr(st, "rerun"):
            st.rerun(scope="app")
            return view
        _render_finale(st, view["finale"]["view"], session_state, on_milestone)
        return view

    if hasattr(st, "markdown"):
        st.markdown(
            build_kpi_band_html(view["evolution"]["detail"]),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="mj-candidate-nav"><span>三条叙事机会</span>'
            '<small>选择一条查看同阶段轨迹与叙事</small></div>',
            unsafe_allow_html=True,
        )
    if hasattr(st, "columns"):
        candidate_columns = st.columns((1, 1, 1), gap="small")
    else:
        candidate_columns = (_NullContext(),) * 3
    label_by_id = {candidate_id: label for candidate_id, label, _ in _CANDIDATES}
    if hasattr(st, "button"):
        for candidate_column, candidate_id in zip(
            candidate_columns, run.selected_candidate_ids, strict=True
        ):
            with candidate_column:
                st.button(
                    label_by_id[candidate_id],
                    key=f"evolution_candidate_{candidate_id}",
                    use_container_width=True,
                    disabled=candidate_id == selected_candidate_id,
                    on_click=apply_playback_action,
                    args=(session_state, PlaybackAction.SELECT_CANDIDATE),
                    kwargs={
                        "candidate_id": candidate_id,
                        "on_milestone": on_milestone,
                    },
                )

    if hasattr(st, "columns"):
        stage_columns = st.columns((2.15, 1), gap="large")
        main_column, right_column = stage_columns
    else:
        main_column = right_column = _NullContext()
    phase_labels = {
        PlaybackPhase.COMMENTS: ("01", "评论进入", "五条完整评论正在进入本轮判断"),
        PlaybackPhase.SCORE: ("02", "数据更新", "新线段、分数与排名在此刻出现"),
        PlaybackPhase.NARRATIVE: ("03", "叙事更新", "旧文本淡出，新文本按真实修改渐入"),
    }
    if state.checkpoint_index == 0 and not state.finale:
        phase_number, phase_title, phase_note = (
            "00",
            "待启动",
            "AI 原始锚点已冻结；点击「下一步」开始实时演化回放",
        )
    else:
        phase_number, phase_title, phase_note = phase_labels[state.phase]

    with main_column:
        if view["evolution"]["ticker_html"] and hasattr(st, "markdown"):
            st.markdown(view["evolution"]["ticker_html"], unsafe_allow_html=True)
        elif (
            state.phase is PlaybackPhase.COMMENTS
            and state.checkpoint_index == 1
            and hasattr(st, "markdown")
        ):
            st.markdown(
                '<section class="mj-blind-confirmation"><span>盲评确认进入</span>'
                '<h3>三条机会由 AI 推荐，经真人盲评验证与团队确认后进入实时演化</h3>'
                '<p>这一轮不追加评论，只建立盲评前后选择的共同起点。</p></section>',
                unsafe_allow_html=True,
            )
        elif hasattr(st, "markdown"):
            idle_text = (
                "AI 原始锚点已冻结，点击下一步进入盲评后重评"
                if state.checkpoint_index == 0
                else "本批评论已进入，继续推进折线与叙事"
            )
            st.markdown(
                f'<div class="mj-comment-idle">{idle_text}</div>',
                unsafe_allow_html=True,
            )
        if hasattr(st, "markdown"):
            st.markdown(
                f"<div class='mj-live-panel' data-visible-points='{view['evolution']['visible_checkpoint_index'] + 1}'>"
                "<span class='mj-live-kicker'>评分轨迹 · 逐点揭示</span></div>",
                unsafe_allow_html=True,
            )
            if (
                state.phase is PlaybackPhase.SCORE
                and view["evolution"]["visible_checkpoint_index"] > 0
            ):
                st.markdown(
                    f"<style>{LATEST_SEGMENT_ANIMATION_CSS}</style>",
                    unsafe_allow_html=True,
                )
        if hasattr(st, "plotly_chart"):
            selection = st.plotly_chart(
                view["evolution"]["figure"],
                use_container_width=True,
                key="evolution_score_chart",
                on_select="rerun",
                selection_mode="points",
            )
        else:
            selection = None

    marker = _selected_marker(selection)
    if marker is not None:
        if marker[0] not in run.selected_candidate_ids:
            raise ValueError(f"marker candidate 不在正式演化 view: {marker[0]}")
        apply_playback_action(
            session_state,
            PlaybackAction.SELECT_CANDIDATE,
            candidate_id=marker[0],
            on_milestone=on_milestone,
        )
        state = _playback_state_from_session(session_state)
        selected_candidate_id = marker[0]
        view = build_dashboard_view(
            run,
            state,
            selected_candidate_id=selected_candidate_id,
        )

    full_width_detail: Mapping[str, Any] | None = None
    with right_column:
        if view["evolution"]["detail_error"] and hasattr(st, "error"):
            st.error(view["evolution"]["detail_error"])
        elif view["evolution"]["detail"] is not None and hasattr(st, "markdown"):
            detail = view["evolution"]["detail"]
            narrative_index = (
                state.checkpoint_index
                if state.phase is PlaybackPhase.NARRATIVE or state.checkpoint_index == 0
                else state.checkpoint_index - 1
            )
            narrative_detail = build_checkpoint_detail(
                run,
                checkpoint_index=narrative_index,
                candidate_id=selected_candidate_id,
            )
            st.markdown(
                build_narrative_transition_html(
                    narrative_detail,
                    animate=state.phase is PlaybackPhase.NARRATIVE,
                ),
                unsafe_allow_html=True,
            )
            full_width_detail = detail
    if hasattr(st, "columns"):
        control_columns = st.columns((1, 1, 1, 5), gap="small")
    else:
        control_columns = (_NullContext(),) * 4
    actions = (
        ("上一步", PlaybackAction.PREVIOUS, "evolution_previous"),
        ("下一步", PlaybackAction.NEXT_CHANGE, "evolution_next"),
        ("从头重播", PlaybackAction.REPLAY, "evolution_replay"),
    )
    if hasattr(st, "button"):
        for control_column, (label, action, key) in zip(
            control_columns[:3], actions, strict=True
        ):
            with control_column:
                st.button(
                    label,
                    key=key,
                    use_container_width=True,
                    on_click=apply_playback_action,
                    args=(session_state, action),
                    kwargs={"on_milestone": on_milestone},
                )
    if hasattr(st, "markdown"):
        st.markdown(
            build_progress_strip_html(view["evolution"]["nodes"], state),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<section class="mj-live-phase">'
            f'<span>{phase_number} / 03</span><strong>{phase_title}</strong>'
            f'<p>{phase_note}</p></section>',
            unsafe_allow_html=True,
        )
    if full_width_detail is not None and hasattr(st, "markdown"):
        st.markdown(
            build_full_width_detail_html(full_width_detail),
            unsafe_allow_html=True,
        )
    render_scroll_continuity(
        f"realtime:{state.checkpoint_index}:{state.phase.value}"
    )
    return view


@_streamlit.fragment
def _render_live_streamlit_fragment(
    root: Any,
    on_milestone: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Stable fragment identity for Streamlit and AppTest widget events."""

    return render_realtime_decision_dashboard(
        root,
        streamlit_module=_streamlit,
        _fragment_body=True,
        on_milestone=on_milestone,
    )


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _render_finale(
    st: Any,
    finale: FinaleView,
    session_state: MutableMapping[str, object],
    on_milestone: Callable[[], None] | None = None,
) -> None:
    """Render the rose finale from the frozen synthesis only."""

    act = int(session_state.get("evolution_finale_act", 1) or 1)
    if act not in {1, 2}:
        raise ValueError("终幕 act 只能是 1 或 2")
    risk_markup = "".join(f"<li>{escape(item)}</li>" for item in finale.remaining_risks)
    forbidden_markup = "".join(
        f"<li>{escape(item)}</li>" for item in finale.forbidden_claims
    )
    scene_markup = "".join(
        '<article><span>'
        f"{escape(scene.level)}</span><h3>{escape(scene.title)}</h3>"
        f"<p>{escape(scene.body)}</p></article>"
        for scene in finale.scenes
    )
    if hasattr(st, "markdown"):
        st.markdown(
            f"<style>{FINALE_PAGE_CSS}</style>" + build_finale_html(finale, act=act),
            unsafe_allow_html=True,
        )
    if hasattr(st, "button"):
        if act == 1:
            if st.button(
                "下一步",
                key="evolution_finale_expand",
                use_container_width=True,
                type="primary",
            ):
                session_state["evolution_finale_act"] = 2
                session_state["evolution_finale_auto_reveal"] = False
                if on_milestone is not None:
                    on_milestone()
                if hasattr(st, "rerun"):
                    st.rerun()
        else:
            controls = st.columns(2) if hasattr(st, "columns") else (None, None)
            with controls[0] if controls[0] is not None else _NullContext():
                if st.button("上一步", key="evolution_finale_previous", use_container_width=True):
                    session_state["evolution_finale_act"] = 1
                    if on_milestone is not None:
                        on_milestone()
                    if hasattr(st, "rerun"):
                        st.rerun()
            with controls[1] if controls[1] is not None else _NullContext():
                if st.button("从头重播", key="evolution_finale_replay", use_container_width=True):
                    apply_playback_action(
                        session_state, PlaybackAction.REPLAY, on_milestone=on_milestone
                    )
                    if hasattr(st, "rerun"):
                        st.rerun()
    if act == 2 and hasattr(st, "expander"):
        with st.expander("完整方案 · 场景、边界与证据", expanded=False):
            if hasattr(st, "write"):
                st.write(finale.decision_brief or finale.core_narrative)
            if hasattr(st, "markdown"):
                st.markdown(
                    f"<div class='mj-finale-drawer-scenes'>{scene_markup}</div>"
                    "<div class='mj-finale-guardrails'>"
                    f"<section><h3>仍需留意</h3><ul>{risk_markup}</ul></section>"
                    f"<section><h3>不可宣称</h3><ul>{forbidden_markup}</ul></section>"
                    "</div>",
                    unsafe_allow_html=True,
                )


def _playback_state_from_session(session_state: MutableMapping[str, object]) -> PlaybackState:
    raw_phase = session_state.get("evolution_phase", PlaybackPhase.SCORE.value)
    try:
        phase = PlaybackPhase(raw_phase)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"未知实时播放子阶段：{raw_phase}") from exc
    return PlaybackState(
        checkpoint_index=session_state.get("evolution_checkpoint_index", 0),
        phase=phase,
        playing=session_state.get("evolution_playing", False),
        finale=session_state.get("evolution_show_finale", False),
    )


def _selected_marker(selection: Any) -> list[Any] | None:
    if selection is None:
        return None
    points = (
        selection.get("points")
        if isinstance(selection, dict)
        else getattr(selection, "points", None)
    )
    if not points:
        return None
    first = points[0]
    customdata = (
        first.get("customdata")
        if isinstance(first, dict)
        else getattr(first, "customdata", None)
    )
    if not isinstance(customdata, (tuple, list)) or len(customdata) != 2:
        return None
    if not isinstance(customdata[0], str) or not isinstance(customdata[1], int):
        return None
    return [customdata[0], customdata[1]]


def build_score_figure(
    run: Any,
    selected_candidate_id: str,
    *,
    visible_checkpoint_index: int = _NUMERIC_NODE_COUNT - 1,
    animate_latest_segment: bool = False,
) -> go.Figure:
    """Build the six-node, three-candidate score figure from the read-only view."""

    if selected_candidate_id not in _CANDIDATE_IDS:
        raise ValueError(f"未知候选 ID: {selected_candidate_id}")
    selected_ids = getattr(run, "selected_candidate_ids", None)
    if not isinstance(selected_ids, (tuple, list)) or set(selected_ids) != _CANDIDATE_IDS:
        raise ValueError("正式演化 view 的候选 ID 必须与三条固定候选线一致")
    points = getattr(run, "numeric_points", None)
    if not isinstance(points, (tuple, list)) or len(points) != 6:
        raise ValueError("正式演化 view 必须包含六个 numeric_points")
    if (
        not isinstance(visible_checkpoint_index, int)
        or isinstance(visible_checkpoint_index, bool)
        or not 0 <= visible_checkpoint_index < len(points)
    ):
        raise ValueError("visible_checkpoint_index 必须在 0 到 5 之间")

    stage_labels = [_display_stage_label(index) for index in range(len(points))]
    all_scores: list[float] = []
    for point in points:
        candidates = getattr(point, "candidates", None)
        if not isinstance(candidates, (tuple, list)):
            raise ValueError("numeric point 缺少 candidates")
        by_id = {getattr(item, "candidate_id", None): item for item in candidates}
        for candidate_id in _CANDIDATE_IDS:
            candidate = by_id.get(candidate_id)
            score = getattr(candidate, "weighted_score", None)
            if not isinstance(score, (int, float)):
                raise ValueError(f"正式演化 view 缺少有效分数：{candidate_id}")
            all_scores.append(float(score))
    score_range = [min(all_scores) - 1.5, max(all_scores) + 1.5]

    figure = go.Figure()
    latest_segments: list[
        tuple[
            str,
            str,
            str,
            list[int],
            list[str],
            list[float],
            list[list[Any]],
            bool,
        ]
    ] = []
    label_positions = ("top right", "middle right", "bottom right")
    for candidate_index, (candidate_id, label, color) in enumerate(_CANDIDATES):
        x_values: list[int] = []
        stage_text_values: list[str] = []
        y_values: list[float] = []
        customdata: list[list[Any]] = []
        for checkpoint_index, point in enumerate(points[: visible_checkpoint_index + 1]):
            point_id = getattr(point, "checkpoint_id", None)
            candidates = getattr(point, "candidates", None)
            if not isinstance(point_id, str) or not isinstance(candidates, (tuple, list)):
                raise ValueError("numeric point 缺少有效 checkpoint_id 或 candidates")
            candidate = next(
                (item for item in candidates if getattr(item, "candidate_id", None) == candidate_id),
                None,
            )
            if candidate is None:
                raise ValueError(f"检查点 {point_id} 缺少候选 {candidate_id}")
            score = getattr(candidate, "weighted_score", None)
            if not isinstance(score, (int, float)):
                raise ValueError(f"检查点 {point_id} 的 {candidate_id} 缺少有效 weighted_score")
            x_values.append(checkpoint_index)
            stage_text_values.append(_display_stage_label(checkpoint_index))
            y_values.append(float(score))
            customdata.append([candidate_id, checkpoint_index])

        is_selected = candidate_id == selected_candidate_id
        has_latest_segment = visible_checkpoint_index > 0
        base_x = x_values[:-1] if has_latest_segment else x_values
        base_text = stage_text_values[:-1] if has_latest_segment else stage_text_values
        base_y = y_values[:-1] if has_latest_segment else y_values
        base_customdata = customdata[:-1] if has_latest_segment else customdata
        base_labels = ["" for _ in base_x]
        if not has_latest_segment and base_labels:
            base_labels[-1] = f"{label}  {base_y[-1]:.2f}"
        figure.add_trace(
            go.Scatter(
                x=base_x,
                text=base_labels,
                hovertext=base_text,
                y=base_y,
                mode="lines+markers+text",
                name=label,
                uid=f"score-history-{candidate_id}",
                meta="settled-history",
                opacity=1.0 if is_selected else 0.35,
                line={"color": color, "width": 4 if is_selected else 2},
                marker={
                    "color": color,
                    "size": [
                        (8 if is_selected else 6)
                        for _ in base_x
                    ],
                },
                textposition=label_positions[candidate_index],
                textfont={"color": color, "size": 11},
                cliponaxis=False,
                customdata=base_customdata,
                hovertemplate=(
                    f"{label}<br>阶段=%{{hovertext}}<br>加权分=%{{y:.2f}}<extra></extra>"
                ),
            )
        )
        if has_latest_segment:
            latest_segments.append(
                (
                    candidate_id,
                    label,
                    color,
                    x_values[-2:],
                    stage_text_values[-2:],
                    y_values[-2:],
                    customdata[-2:],
                    is_selected,
                )
            )
        else:
            latest_segments.append(
                (candidate_id, label, color, [], [], [], [], is_selected)
            )

    for latest_index, (
        candidate_id,
        label,
        color,
        x_values,
        stage_text_values,
        y_values,
        customdata,
        is_selected,
    ) in enumerate(latest_segments):
        figure.add_trace(
            go.Scatter(
                x=x_values,
                text=["", f"{label}  {y_values[-1]:.2f}"] if y_values else [],
                hovertext=stage_text_values,
                y=y_values,
                mode="lines+markers+text",
                name=f"{label} · 新增",
                uid=f"score-latest-{candidate_id}",
                meta="latest-segment",
                showlegend=False,
                opacity=1.0 if is_selected else 0.35,
                line={"color": color, "width": 4 if is_selected else 2},
                marker={"color": color, "size": [0, 12 if is_selected else 8]},
                textposition=label_positions[latest_index],
                textfont={"color": color, "size": 11},
                cliponaxis=False,
                customdata=customdata,
                hovertemplate=(
                    f"{label}<br>阶段=%{{hovertext}}<br>加权分=%{{y:.2f}}<extra></extra>"
                ),
            )
        )

    visible_tickvals = list(range(visible_checkpoint_index + 1))
    visible_ticktext = stage_labels[: visible_checkpoint_index + 1]
    if visible_checkpoint_index < len(stage_labels) - 1:
        visible_tickvals.append(visible_checkpoint_index + 1)
        visible_ticktext.append("下一变化待定")
    axis_right = max(0.85, visible_tickvals[-1] + 0.15)

    figure.update_layout(
        xaxis_title="实时变化阶段",
        yaxis_title="加权分",
        hovermode="closest",
        showlegend=False,
        height=320,
        paper_bgcolor="#111B1F",
        plot_bgcolor="#111B1F",
        font={"color": "#CDD4D3", "family": "Microsoft YaHei"},
        margin={"l": 50, "r": 72, "t": 34, "b": 48},
        uirevision="meijian-score-trajectory",
        meta={
            "animate_latest_segment": bool(
                animate_latest_segment and visible_checkpoint_index > 0
            )
        },
        xaxis={
            "gridcolor": "rgba(148,163,167,.12)",
            "zeroline": False,
            "type": "linear",
            "tickmode": "array",
            "tickvals": visible_tickvals,
            "ticktext": visible_ticktext,
            "range": [-0.15, axis_right],
        },
        yaxis={
            "gridcolor": "rgba(148,163,167,.12)",
            "zeroline": False,
            "range": score_range,
        },
    )
    return figure


def build_ticker_html(batch: Any) -> str:
    """Render five complete comments across three staggered moving lanes."""

    batch_id = getattr(batch, "batch_id", None)
    records = getattr(batch, "records", None)
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("delta batch 缺少有效 batch_id")
    if not isinstance(records, (tuple, list)) or len(records) != 5:
        raise ValueError("评论回放批次必须包含五条记录")

    for record in records:
        for field_name in ("raw_id", "raw_content", "source_platform", "source_ref"):
            value = getattr(record, field_name, None)
            if not isinstance(value, str) or not value:
                raise ValueError(f"评论回放记录缺少有效 {field_name}")

    total_characters = sum(len(record.raw_content) for record in records)
    duration_seconds = max(20, min(44, ceil(total_characters / 10)))
    lanes: list[list[str]] = [[], [], []]
    for index, record in enumerate(records):
        platform_view = _PLATFORM_AVATARS.get(record.source_platform)
        if platform_view is None:
            raise ValueError(f"评论平台没有已批准的头像样式: {record.source_platform}")
        platform_class, platform_mark = platform_view
        lane_index = (0, 1, 2, 0, 1)[index]
        lanes[lane_index].append(
            f'<span class="mj-comment-bullet is-item-{index}">'
            f'<span class="mj-platform-avatar is-{platform_class}">{escape(platform_mark)}</span>'
            '<span class="mj-comment-copy">'
            f'<strong>{escape(record.source_platform)}</strong>'
            f'<span>{escape(record.raw_content)}</span></span></span>'
        )
    lane_markup = "".join(
        f'<div class="mj-comment-lane">{"".join(items)}</div>' for items in lanes
    )
    static_items = "".join(
        '<li class="mj-evidence-ticker__record">'
        f'<span class="mj-evidence-ticker__platform">{escape(record.source_platform)}</span>'
        f'<p>{escape(record.raw_content)}</p>'
        '</li>'
        for record in records
    )
    return (
        f"<section class=\"mj-evidence-ticker\" data-batch-id=\"{escape(batch_id)}\" "
        f"data-record-count=\"{len(records)}\" data-duration-seconds=\"{duration_seconds}\">"
        '<h3>新增评论实时进入</h3>'
        f'<div class="mj-comment-stage">{lane_markup}</div>'
        f'<footer class="mj-comment-complete">本批 {len(records)} 条正在进入</footer>'
        f'<details><summary>查看本批 {len(records)} 条完整评论</summary><ol>{static_items}</ol></details>'
        "</section>"
    )


def attribution_projection(
    run: Any,
    checkpoint_index: int,
    candidate_id: str | None = None,
) -> AttributionProjection:
    """Expose only recorded patch/risk/scene/brief fields for one checkpoint.

    The current upstream view does not yet carry checkpoint scenes and briefs;
    that absence is surfaced explicitly instead of borrowing final-synthesis data.
    """

    if not isinstance(checkpoint_index, int) or isinstance(checkpoint_index, bool):
        raise ValueError("checkpoint_index 必须是整数")
    points = getattr(run, "numeric_points", None)
    if not isinstance(points, (tuple, list)) or not 0 <= checkpoint_index < len(points):
        raise ValueError("checkpoint_index 不在正式 numeric_points 范围内")
    point = points[checkpoint_index]
    missing = [
        field_name
        for field_name in ("scenes", "decision_brief")
        if not hasattr(point, field_name)
    ]
    if missing:
        raise ValueError(f"checkpoint attribution 缺少字段: {', '.join(missing)}")

    scenes = getattr(point, "scenes")
    decision_brief = getattr(point, "decision_brief")
    if not isinstance(scenes, (tuple, list)) or not isinstance(decision_brief, str):
        raise ValueError("checkpoint scenes 或 decision_brief 类型无效")
    if not all(isinstance(scene, dict) for scene in scenes):
        raise ValueError("checkpoint scenes 必须是对象列表")

    candidates = getattr(point, "candidates", None)
    if not isinstance(candidates, (tuple, list)):
        raise ValueError("numeric point 缺少 candidates")
    if candidate_id is None:
        candidate_id = getattr(run, "selected_candidate_ids", [None])[0]
    candidate = next(
        (item for item in candidates if getattr(item, "candidate_id", None) == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"检查点缺少候选 {candidate_id}")
    patches = getattr(candidate, "patches", None)
    risks = getattr(candidate, "risks", None)
    if not isinstance(patches, (tuple, list)) or not isinstance(risks, (tuple, list)):
        raise ValueError("checkpoint candidate 缺少 patches 或 risks")
    if not all(isinstance(patch, dict) for patch in patches):
        raise ValueError("checkpoint patches 必须是对象列表")
    if not all(isinstance(risk, str) for risk in risks):
        raise ValueError("checkpoint risks 必须是字符串列表")

    checkpoint_id = getattr(point, "checkpoint_id", None)
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("numeric point 缺少有效 checkpoint_id")
    return AttributionProjection(
        checkpoint_index=checkpoint_index,
        checkpoint_id=checkpoint_id,
        candidate_id=candidate_id,
        patches=tuple(patches),
        risks=tuple(risks),
        scenes=tuple(scenes),
        decision_brief=decision_brief,
    )


def _render_record_link(record: Any) -> str:
    url = getattr(record, "original_url", None)
    if not isinstance(url, str) or not url:
        return "无可用链接"
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"original_url 必须是 HTTP(S) 链接: {record.raw_id}")
    safe_url = escape(url, quote=True)
    return f'<a href="{safe_url}" target="_blank" rel="noreferrer">打开原文</a>'


__all__ = [
    "AttributionProjection",
    "FinaleSceneView",
    "FinaleView",
    "PlaybackAction",
    "PlaybackPhase",
    "PlaybackState",
    "apply_playback_action",
    "attribution_projection",
    "build_checkpoint_detail",
    "build_attribution_strip_html",
    "build_dashboard_view",
    "build_detail_html",
    "build_dimension_tab_html",
    "build_evidence_risk_tab_html",
    "build_finale_view",
    "build_finale_html",
    "build_full_width_detail_html",
    "build_narrative_transition_html",
    "build_progress_strip_html",
    "build_score_figure",
    "build_score_summary_html",
    "build_scene_tab_html",
    "build_ticker_html",
    "initialize_realtime_session",
    "reduce_playback",
    "render_realtime_decision_dashboard",
    "LIVE_PAGE_CSS",
    "FINALE_PAGE_CSS",
]
