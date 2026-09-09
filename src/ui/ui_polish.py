"""UI polish overlay (ui branch).

即插即用美化层：全部新增样式集中在本模块，由 app.py 在主题之后注入；
移除 app.py 中对 build_polish_css 的注入即可整体回退到原观感。
按板块分段（global / gateway / preprocessing / pressure / dashboard / finale），
便于逐板块禁用或调整。仅含视觉层，不改任何业务行为。
"""

from __future__ import annotations


def build_system_rail_html(active_workspace: str) -> str:
    """Non-interactive brand rail; navigation stays in native controls."""

    return (
        '<div class="mj-system-rail" aria-hidden="true">'
        '<svg class="mj-rail-mark" viewBox="0 0 100 100">'
        '<circle cx="50" cy="26" r="16"/><circle cx="72.8" cy="42.6" r="16"/>'
        '<circle cx="64.1" cy="69.4" r="16"/><circle cx="35.9" cy="69.4" r="16"/>'
        '<circle cx="27.2" cy="42.6" r="16"/>'
        '<circle class="mj-rail-core" cx="50" cy="50" r="4.2"/></svg>'
        '<span class="mj-rail-brand">梅见</span></div>'
    )


def build_status_bar_html(active_workspace: str) -> str:
    """Fixed bottom status bar with real session facts only."""

    return (
        '<div class="mj-status-bar" aria-hidden="true">'
        "<span>梅见 · 叙事决策工作台</span>"
        f"<b>当前模块 · {active_workspace}</b>"
        "</div>"
    )


def build_polish_css() -> str:
    return """
/* ============ polish: global tokens ============ */
:root{
  --mj-ease-out-expo:cubic-bezier(.16,1,.3,1);
  --mj-ease-curtain:cubic-bezier(.65,0,.35,1);
  --mj-ease-back:cubic-bezier(.34,1.56,.64,1);
  --mj-polish-wine:#8F2F4D;
  --mj-polish-gold:#C9A86A;
  --mj-polish-ink:#352F2B;
}
@keyframes mjPolishRise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes mjPolishDraw{from{stroke-dashoffset:var(--mj-len,320)}to{stroke-dashoffset:0}}
@keyframes mjPolishLineIn{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes mjPolishFade{from{opacity:0}to{opacity:1}}
@keyframes mjLedBreath{0%,100%{box-shadow:0 0 0 0 rgba(159,224,209,.35)}50%{box-shadow:0 0 0 .38rem rgba(159,224,209,.07)}}
@keyframes mjStageTrackFill{from{transform:scaleX(0);opacity:.35}to{transform:scaleX(1);opacity:1}}
@keyframes mjStageTrackFlow{from{background-position:-9rem 0}to{background-position:9rem 0}}
@keyframes mjValidateCheck{from{opacity:0;transform:translateY(-.25rem)}to{opacity:1;transform:translateY(0)}}
@keyframes mjCleanBranch{from{opacity:0;transform:translateX(-.65rem)}to{opacity:1;transform:translateX(0)}}
@keyframes mjSplitLane{from{opacity:0;transform:scaleX(.2);transform-origin:left center}to{opacity:1;transform:scaleX(1);transform-origin:left center}}
@keyframes mjAnnotateField{from{opacity:0;transform:translateY(.35rem)}to{opacity:1;transform:translateY(0)}}
@keyframes mjFreezeMerge{from{opacity:0;transform:translateX(-.45rem)}to{opacity:1;transform:translateX(0)}}
@media (prefers-reduced-motion: reduce){
  .mj-system-gateway *,.mj-polish-anim *{animation-duration:.01ms!important;animation-delay:0ms!important;transition-duration:.01ms!important}
  .mj-stage-track *, .mj-stage-workbench *{animation:none!important;transition:none!important}
}

/* ============ polish: gateway ============ */
body:has(.mj-system-gateway) [data-testid="stAppViewContainer"]{background:#F6F5F2}
body:has(.mj-system-gateway) .block-container{max-width:90rem;padding-left:3vw;padding-right:3vw}
.mj-system-gateway__head{animation:mjPolishRise .45s var(--mj-ease-out-expo) both}
[data-testid="column"]:has(.mj-system-gateway) [data-testid="stButton"]{max-width:27rem;margin:0 0 .65rem}
[data-testid="column"]:has(.mj-system-gateway) [data-testid="stButton"]>button{border:1px solid #BEA5AD;border-radius:7px;min-height:2.85rem;color:#8F2F4D;background:transparent;box-shadow:none;transition:background .18s,color .18s}
[data-testid="column"]:has(.mj-system-gateway) [data-testid="stButton"]>button[kind="primary"]{background:#8F2F4D;border-color:#8F2F4D;color:#FFF8EF}
[data-testid="column"]:has(.mj-system-gateway) [data-testid="stButton"]>button:hover{background:#78273F;color:#FFF8EF;border-color:#78273F}
[data-testid="column"]:has(.mj-system-gateway) [data-testid="stButton"]>button p{font-size:.88rem;letter-spacing:.05em}

/* ============ polish: preprocessing ============ */
.mj-preprocess-sourcebar{display:grid;grid-template-columns:auto minmax(10rem,1fr) auto auto auto;align-items:center;gap:.7rem;margin:.45rem 0 .8rem;padding:.5rem 0;border-bottom:1px solid #DED3C8;color:#736861;font-size:.72rem;animation:mjPolishRise .36s var(--mj-ease-out-expo) both}
.mj-sourcebar-label{color:var(--mj-polish-wine);font-size:.64rem;font-weight:750;letter-spacing:.15em}.mj-preprocess-sourcebar strong{color:#3F3036;font-family:STZhongsong,"华文中宋",serif;font-size:.94rem;font-weight:600}.mj-preprocess-sourcebar b{padding-left:.7rem;border-left:1px solid #D8CCBE;color:#8F2F4D;font-family:Georgia,serif;font-size:.88rem;font-variant-numeric:tabular-nums}.mj-preprocess-sourcebar small{color:#91857C;font-size:.63rem}.mj-preprocess-sourcebar--custom{margin-top:.25rem}
.mj-stage-track{position:relative;margin:.8rem 0 1.15rem;padding:.1rem 0}.mj-stage-track ol{position:relative;z-index:1;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:0;margin:0;padding:0;list-style:none}.mj-stage-rail{position:absolute;z-index:0;left:10%;right:10%;top:.58rem;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));align-items:center}.mj-stage-connector{position:relative;height:5px;margin:0 .15rem;overflow:hidden;border-radius:99px;background:#DDD4CB}.mj-stage-connector::before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,#C9A86A 0 72%,#F0DDB0 86%,#C9A86A 100%);background-size:9rem 100%;transform:scaleX(0);transform-origin:left center}.mj-stage-connector.is-complete::before{transform:scaleX(1);animation:mjStageTrackFill .78s var(--mj-ease-curtain) both}.mj-stage-connector.is-flowing::before{transform:scaleX(1);animation:mjStageTrackFlow .82s linear both}.mj-stage-node{position:relative;display:grid;grid-template-columns:1fr;grid-template-rows:1.2rem auto auto;justify-items:center;row-gap:.12rem;align-items:center;min-width:0;color:#988D85}.mj-stage-dot{position:relative;z-index:2;grid-row:1;grid-column:1;display:grid;place-items:center;width:.98rem;height:.98rem;border:1px solid #CFC2B6;border-radius:50%;background:#F6F5F2;color:#8C8178;font-family:Georgia,serif;font-size:.56rem}.mj-stage-node strong{grid-row:2;grid-column:1;margin-top:.15rem;overflow:hidden;max-width:100%;color:#5B514B;font-family:STZhongsong,"华文中宋",serif;font-size:.82rem;text-align:center;text-overflow:ellipsis;white-space:nowrap}.mj-stage-node small{grid-row:3;grid-column:1;font-size:.61rem;text-align:center}.mj-stage-node.is-complete .mj-stage-dot{border-color:#C9A86A;background:#C9A86A;color:#FFF8EF}.mj-stage-node.is-current .mj-stage-dot{border:2px solid var(--mj-polish-wine);box-shadow:0 0 0 3px rgba(143,47,77,.1);color:var(--mj-polish-wine)}.mj-stage-node.is-current strong{color:var(--mj-polish-wine)}.mj-stage-node.is-blocked{opacity:.52}.mj-stage-seal{position:absolute;right:.15rem;top:-.75rem;padding:.08rem .28rem;border:1px solid rgba(143,47,77,.5);color:var(--mj-polish-wine);font-family:STZhongsong,"华文中宋",serif;font-size:.56rem;letter-spacing:.08em;transform:rotate(-5deg)}
.mj-custom-stage-note{grid-column:1/-1;margin:.65rem 0 0;padding:.45rem .65rem;border-left:2px solid var(--mj-polish-wine);background:rgba(255,249,245,.7);color:#675D56;font-size:.72rem}.mj-custom-stage-note strong{color:#8F2F4D}.mj-custom-stage-note p{display:inline;margin:0 0 0 .45rem}
.mj-stage-workbench{margin:.35rem 0 1rem;padding:1.1rem 1.2rem;border:0;border-left:2px solid var(--mj-polish-wine);border-radius:0;background:#FFFCF7;box-shadow:0 12px 34px rgba(63,48,54,.06);animation:mjPolishRise .42s var(--mj-ease-out-expo) both}.mj-stage-workbench__header{margin-bottom:1rem}.mj-stage-workbench__header span{color:#8F2F4D;font-size:.64rem;font-weight:750;letter-spacing:.14em}.mj-stage-workbench__header h2{font-size:1.35rem}
.mj-stage-workbench article{min-height:0;padding:0;border:0;background:transparent}.mj-stage-workbench article p{margin:.35rem 0;color:#504740;font-size:.8rem;line-height:1.65}.mj-stage-workbench small{color:#8F2F4D;font-size:.63rem;font-weight:750;letter-spacing:.1em}.mj-validate-flow{display:grid;grid-template-columns:1fr 1.8rem 1.15fr 1.8rem 1fr;gap:.65rem;align-items:center}.mj-validate-flow>i,.mj-annotation-transform>i,.mj-freeze-convergence>i{height:1px;background:linear-gradient(90deg,#C9A86A,#D8CCBE)}.mj-flow-check{padding:0 .8rem!important;border-left:1px solid #D8CCBE!important;border-right:1px solid #D8CCBE!important}.mj-flow-check ul{display:flex;flex-wrap:wrap;gap:.3rem;margin:.5rem 0 0;padding:0;list-style:none}.mj-flow-check li{padding:.18rem .32rem;background:#F3EEE7;color:#70655C;font-size:.64rem}.mj-flow-output b{color:#8F2F4D;font-family:Georgia,serif;font-variant-numeric:tabular-nums}
.mj-clean-flow{display:grid;grid-template-columns:minmax(10rem,.72fr) 1.7fr;gap:1.25rem;align-items:center}.mj-clean-input{padding-right:1rem!important;border-right:1px solid #D8CCBE!important}.mj-clean-input strong{display:block;margin:.2rem 0;color:#8F2F4D;font-family:Georgia,serif;font-size:2.1rem;font-variant-numeric:tabular-nums}.mj-clean-branch{display:grid;gap:.42rem;position:relative}.mj-clean-branch::before{content:"";position:absolute;left:-1.25rem;top:50%;width:1.25rem;height:1px;background:#C9A86A}.mj-clean-output{display:grid;grid-template-columns:4.4rem 1fr;align-items:baseline;gap:.5rem;min-height:1.9rem;padding:.35rem .5rem!important;border-left:2px solid #D8CCBE!important}.mj-clean-output--1{border-left-color:#8F2F4D!important}.mj-clean-output strong{color:#4F4640;font-family:Georgia,serif;font-size:1.05rem;font-variant-numeric:tabular-nums}.mj-clean-output--1 strong{font-size:1.45rem;color:#8F2F4D}.mj-conservation{grid-column:1/-1;margin:0!important;color:#7A6D64!important;font-size:.68rem!important}
.mj-split-flow>header{display:flex;align-items:baseline;gap:.65rem;margin-bottom:.7rem}.mj-split-flow>header strong{color:#4B413B;font-size:.82rem}.mj-split-flow>div{display:grid;gap:.45rem}.mj-split-lane{display:grid;grid-template-columns:4.4rem 1fr auto 8rem;gap:.65rem;align-items:center}.mj-split-lane span{color:#6A5E56;font-size:.74rem}.mj-split-lane i{height:1px;background:repeating-linear-gradient(90deg,#C9A86A 0 4px,transparent 4px 7px)}.mj-split-lane strong{color:#8F2F4D;font-family:Georgia,serif;font-size:1rem;font-variant-numeric:tabular-nums}.mj-split-lane small{color:#8A7F77;font-size:.58rem}.mj-split-flow>p{margin:.65rem 0 0;color:#7B7067;font-size:.68rem}
.mj-annotation-transform{display:grid;grid-template-columns:minmax(9rem,1fr) 1.5rem minmax(12rem,1.2fr) minmax(10rem,.85fr);gap:.75rem;align-items:center}.mj-annotation-text{padding-right:.75rem!important;border-right:1px solid #DED3C8!important}.mj-annotation-fields>div{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.35rem;margin-top:.5rem}.mj-annotation-fields span{padding:.38rem .46rem;border-bottom:1px solid #DED3C8;color:#5C514A;font-size:.68rem}.mj-annotation-transform aside{padding-left:.8rem;border-left:1px solid #DED3C8}.mj-annotation-transform aside ul{display:grid;gap:.25rem;margin:.4rem 0 0;padding:0;list-style:none}.mj-annotation-transform aside li{display:flex;justify-content:space-between;gap:.4rem}.mj-annotation-transform aside li strong{color:#8F2F4D;font-family:Georgia,serif;font-size:.76rem;font-variant-numeric:tabular-nums}
.mj-freeze-convergence{display:grid;grid-template-columns:1fr 1.6rem 8rem 1.6rem 1fr;gap:.7rem;align-items:center}.mj-freeze-packages{display:grid;gap:.35rem}.mj-package-line{position:relative;padding:.28rem .45rem;border-left:2px solid #C9A86A;color:#5B514B;font-size:.68rem}.mj-freeze-node{position:relative;display:grid;place-items:center;min-height:4.7rem!important;background:#F4ECE4!important;border:1px solid #B78591!important}.mj-freeze-node strong{color:#8F2F4D;font-family:STZhongsong,"华文中宋",serif;font-size:1.15rem}.mj-freeze-seal{position:absolute;right:.35rem;top:.35rem;padding:.08rem .28rem;border:1px solid rgba(143,47,77,.65);color:#8F2F4D;font-family:STZhongsong,"华文中宋",serif;font-size:.58rem;transform:rotate(-6deg)}.mj-freeze-output{padding-left:.7rem!important;border-left:1px solid #DED3C8!important}
.mj-stage-workbench[data-stage="validate"] .mj-flow-check li{animation:mjValidateCheck .5s var(--mj-ease-out-expo) both}.mj-stage-workbench[data-stage="validate"] .mj-flow-check li:nth-child(2){animation-delay:.12s}.mj-stage-workbench[data-stage="validate"] .mj-flow-check li:nth-child(3){animation-delay:.24s}
.mj-stage-workbench[data-stage="clean"] .mj-clean-output{animation:mjCleanBranch .68s var(--mj-ease-out-expo) both}.mj-stage-workbench[data-stage="clean"] .mj-clean-output--2{animation-delay:.1s}.mj-stage-workbench[data-stage="clean"] .mj-clean-output--3{animation-delay:.2s}
.mj-stage-workbench[data-stage="split"] .mj-split-lane{animation:mjSplitLane .7s var(--mj-ease-out-expo) both}.mj-stage-workbench[data-stage="split"] .mj-split-lane:nth-child(2){animation-delay:.1s}.mj-stage-workbench[data-stage="split"] .mj-split-lane:nth-child(3){animation-delay:.2s}.mj-stage-workbench[data-stage="split"] .mj-split-lane:nth-child(4){animation-delay:.3s}
.mj-stage-workbench[data-stage="annotate"] .mj-annotation-fields span{animation:mjAnnotateField .48s var(--mj-ease-out-expo) both}.mj-stage-workbench[data-stage="annotate"] .mj-annotation-fields span:nth-child(2){animation-delay:.1s}.mj-stage-workbench[data-stage="annotate"] .mj-annotation-fields span:nth-child(3){animation-delay:.2s}.mj-stage-workbench[data-stage="annotate"] .mj-annotation-fields span:nth-child(4){animation-delay:.3s}
.mj-stage-workbench[data-stage="freeze"] .mj-package-line,.mj-stage-workbench[data-stage="freeze"] .mj-freeze-node{animation:mjFreezeMerge .72s var(--mj-ease-out-expo) both}.mj-stage-workbench[data-stage="freeze"] .mj-package-line:nth-child(2){animation-delay:.12s}.mj-stage-workbench[data-stage="freeze"] .mj-package-line:nth-child(3){animation-delay:.24s}.mj-stage-workbench[data-stage="freeze"] .mj-freeze-node{animation-delay:.42s}
.mj-opportunity-ledger{margin:.25rem 0}.mj-opportunity-row{display:grid;grid-template-columns:2.1rem minmax(9rem,.9fr) 4.4rem minmax(14rem,1.8fr) minmax(10rem,1.25fr);gap:.7rem;align-items:start;padding:.8rem 0;border-bottom:1px solid #E2D8CD}.mj-opportunity-rank{color:#A4978E;font-family:Georgia,serif;font-size:.75rem}.mj-opportunity-row h3{margin:0;color:#3F3036;font-family:STZhongsong,"华文中宋",serif;font-size:.96rem;line-height:1.45}.mj-opportunity-row>strong{display:grid;gap:.12rem;color:#8F2F4D;font-family:Georgia,serif;font-size:1.05rem;font-variant-numeric:tabular-nums}.mj-opportunity-row>strong small,.mj-opportunity-row section small{color:#877A72;font-family:"Microsoft YaHei",sans-serif;font-size:.58rem;font-weight:700;letter-spacing:.08em}.mj-opportunity-row section p,.mj-opportunity-row details p{margin:.25rem 0 0;color:#615750;font-size:.7rem;line-height:1.6}.mj-opportunity-row details summary{color:#8F2F4D;font-size:.68rem;cursor:pointer}.mj-opportunity-row details[open]{padding-left:.55rem;border-left:1px solid #D8CCBE}
.mj-replay-controls{margin:.25rem 0 -.1rem;color:#8F2F4D;font-size:.63rem;font-weight:750;letter-spacing:.14em}.mj-replay-controls+div [data-testid="stButton"]>button,[data-testid="stHorizontalBlock"]:has([class*="st-key-official_replay_"]) [data-testid="stButton"]>button{min-height:2.35rem;border:0;border-bottom:1px solid #D8CCBE;border-radius:0;background:transparent;color:#665B54;box-shadow:none;font-size:.76rem;transition:color .18s ease,border-color .18s ease,background .18s ease}.mj-replay-controls+div [data-testid="stButton"]>button:hover,[data-testid="stHorizontalBlock"]:has([class*="st-key-official_replay_"]) [data-testid="stButton"]>button:hover{border-color:#8F2F4D;background:#FFF8F3;color:#8F2F4D}.mj-replay-controls+div [data-testid="stButton"]>button[data-testid="baseButton-primary"],[data-testid="stHorizontalBlock"]:has([class*="st-key-official_replay_"]) [data-testid="stButton"]>button[data-testid="baseButton-primary"]{border-color:#8F2F4D;background:#8F2F4D;color:#FFF8EF}
@media(max-width:760px){.mj-preprocess-sourcebar{grid-template-columns:1fr 1fr}.mj-preprocess-sourcebar strong{grid-column:span 2}.mj-stage-track ol{gap:.35rem}.mj-stage-node{grid-template-columns:1fr;text-align:center}.mj-stage-dot{justify-self:center;grid-row:1}.mj-stage-node small{display:none}.mj-stage-seal{display:none}.mj-validate-flow,.mj-clean-flow,.mj-annotation-transform,.mj-freeze-convergence{grid-template-columns:1fr}.mj-validate-flow>i,.mj-annotation-transform>i,.mj-freeze-convergence>i{width:1px;height:1rem;margin:auto}.mj-flow-check,.mj-clean-input,.mj-annotation-transform aside,.mj-freeze-output,.mj-annotation-text{padding:0 0 .65rem!important;border:0!important;border-bottom:1px solid #DED3C8!important}.mj-clean-branch::before{display:none}.mj-opportunity-row{grid-template-columns:1.8rem 1fr 3.7rem}.mj-opportunity-row section,.mj-opportunity-row details{grid-column:2/-1}}
@media(max-width:760px){[data-testid="stHorizontalBlock"]:has([class*="st-key-official_replay_"]) {flex-wrap:wrap!important}[data-testid="stHorizontalBlock"]:has([class*="st-key-official_replay_"])>[data-testid="column"]{flex:0 0 calc(50% - .25rem)!important;width:calc(50% - .25rem)!important}[data-testid="stHorizontalBlock"]:has([class*="st-key-official_replay_"])>[data-testid="column"]:last-child{display:none}}
body:not(:has(.mj-live-shell)):not(:has(.mj-finale-shell)) [data-testid="stAlert"] [data-testid="stNotification"]{background:#FAF6EC;border:1px solid #C9A86A;box-shadow:none}
body:not(:has(.mj-live-shell)):not(:has(.mj-finale-shell)) [data-testid="stAlert"] [data-testid="stNotification"] *{color:#6B4230}

/* ============ polish: pressure theatre ============ */
@keyframes mjPolishBarIn{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes mjPolishBreath{0%,100%{box-shadow:0 0 0 0 rgba(143,47,77,.35)}50%{box-shadow:0 0 0 .3rem rgba(143,47,77,.12)}}
body .pressure-hero{overflow:hidden}
body .pressure-hero::before{content:"";position:absolute;inset:0;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.55'/%3E%3C/svg%3E");opacity:.05;mix-blend-mode:multiply;pointer-events:none}
body .pressure-hero::after{color:rgba(143,47,77,.11);right:3rem;-webkit-mask-image:linear-gradient(180deg,#000 55%,transparent);mask-image:linear-gradient(180deg,#000 55%,transparent)}
body .pressure-stage-node.is-complete i{border-color:#C9A86A;background:#FAF6EC;color:#8F2F4D}
body .pressure-stage-node.is-complete::after{background:linear-gradient(90deg,rgba(201,168,106,.9),rgba(201,168,106,.3))}
body .pressure-stage-node.is-current i{animation:mjPolishBreath 2.2s ease-in-out infinite}
body .pressure-holdout-bar{border-radius:2px}
body .pressure-holdout-bar i{transform-origin:left;animation:mjPolishBarIn .9s var(--mj-ease-curtain) both}
body .pressure-holdout-bar i.is-support{background:linear-gradient(90deg,#8F2F4D,#A6455F)}
body .pressure-holdout-bar i.is-challenge{background:#C9A86A}
body .blind-review-track{-webkit-mask-image:linear-gradient(90deg,transparent,#000 5%,#000 95%,transparent);mask-image:linear-gradient(90deg,transparent,#000 5%,#000 95%,transparent)}
body .blind-review-item p{box-shadow:0 .3rem .8rem rgba(63,48,54,.09)}
body .pressure-terminal{animation:mjPolishRise .55s var(--mj-ease-out-expo) both}
body .pressure-terminal header{position:relative;overflow:hidden}
body .pressure-terminal header::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#D9C69A,rgba(217,198,154,0));animation:mjPolishLineIn 1s var(--mj-ease-curtain) .25s both}
body .pressure-terminal li{animation:mjPolishRise .5s var(--mj-ease-out-expo) both}
body .pressure-terminal li:nth-child(1){animation-delay:.3s}
body .pressure-terminal li:nth-child(2){animation-delay:.42s}
body .pressure-terminal li:nth-child(3){animation-delay:.54s}
body:has(.pressure-stage-bar) [data-testid="stButton"]>button{background:#FFFDF9;border:1px solid #D8CCBE;color:#5A5049;border-radius:.35rem;height:2.6rem;font-size:.8rem;box-shadow:none;transition:border-color .2s ease-out,color .2s ease-out,transform .2s ease-out}
body:has(.pressure-stage-bar) [data-testid="stButton"]>button:hover{border-color:var(--mj-polish-wine);color:var(--mj-polish-wine);transform:translateY(-1px)}
body:has(.pressure-stage-bar) [data-testid="stButton"]>button[data-testid="baseButton-primary"]{background:var(--mj-polish-wine);border-color:var(--mj-polish-wine);color:#FFF8EF}
body:has(.pressure-stage-bar) [data-testid="stButton"]>button[data-testid="baseButton-primary"]:hover{background:#7A2740;color:#FFF8EF}

/* ============ polish: system chrome (rail + status bar) ============ */
.mj-system-rail{position:fixed;left:0;top:0;bottom:2rem;width:3.4rem;z-index:80;display:flex;flex-direction:column;align-items:center;gap:1.15rem;padding:1rem 0 .9rem;background:rgba(13,20,23,.91);border-right:1px solid #26363B;pointer-events:none}
.mj-system-rail::after{content:"";position:absolute;top:0;right:-1px;bottom:0;width:1px;background:linear-gradient(180deg,rgba(201,168,106,.55),rgba(201,168,106,.12));pointer-events:none}
.mj-rail-mark{width:1.55rem;opacity:.9}
.mj-rail-mark circle{fill:none;stroke:#C9A86A;stroke-width:2.4}
.mj-rail-mark .mj-rail-core{fill:#C9A86A;stroke:none}
.mj-rail-brand{writing-mode:vertical-rl;color:#C9A86A;letter-spacing:.3em;font-family:STZhongsong,"华文中宋",serif;font-size:.78rem;margin-top:.7rem}
.mj-status-bar{position:fixed;left:3.4rem;right:0;bottom:0;height:2rem;z-index:79;display:flex;align-items:center;justify-content:space-between;padding:0 1.15rem;background:rgba(13,20,23,.95);border-top:1px solid #26363B;color:#66777B;font-size:.62rem;letter-spacing:.16em;pointer-events:none}
.mj-status-bar b{color:#9FB3AE;font-weight:600;letter-spacing:.18em}
body:has(.mj-system-rail) section.main{padding-left:3.4rem;padding-bottom:2.1rem}
@media(max-width:1100px){.mj-system-rail,.mj-status-bar{display:none}body:has(.mj-system-rail) section.main{padding-left:0;padding-bottom:0}}

/* ============ polish: compact top chrome ============ */
[data-testid="stHeader"]{display:none}
[data-testid="stStatusWidget"]{display:none!important}
.block-container,[data-testid="stAppViewBlockContainer"],[data-testid="stMainBlockContainer"]{padding-top:.45rem!important}
/* Style-only markdown must not leave empty flex rows above the application. */
[data-testid="element-container"]:has([data-testid="stMarkdownContainer"] > style):not(:has([data-testid="stMarkdownContainer"] > :not(style))){display:none}
[data-testid="element-container"]:has(.mj-system-rail),[data-testid="element-container"]:has(.mj-status-bar){position:absolute;height:0;margin:0}
.mj-workspace-nav{display:none}
[data-testid="element-container"]:has(.mj-workspace-nav){display:none}
[data-testid="element-container"]:has(.mj-workspace-nav)+[data-testid="stHorizontalBlock"]{gap:.45rem;margin:0 0 .35rem}
[data-testid="element-container"]:has(.mj-workspace-nav)+[data-testid="stHorizontalBlock"] button{height:2.25rem;min-height:2.25rem;font-size:.75rem}
/* This row contains only the shared header, Feishu popover and entry switch. */
[data-testid="stHorizontalBlock"]:has(.mj-system-header){align-items:center;gap:.75rem;min-height:60px;border-bottom:1px solid #DDD6D0;margin-bottom:.2rem;position:relative}
[data-testid="stHorizontalBlock"]:has(.mj-system-header)>[data-testid="column"]{position:static}
[data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stPopover"]{margin:0;background:transparent}
[data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stPopover"]>button{min-height:2.6rem;padding:.55rem .7rem;border:1px solid #D8CCBE;border-radius:7px;background:#FFFDF9;color:#5B514B;font-size:.76rem;white-space:nowrap}
[data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stPopover"]>button:hover{border-color:#8F2F4D;color:#8F2F4D}
[data-testid="stPopoverBody"]{width:min(34rem,85vw)!important;padding:1rem 1.1rem!important;background:#F6F5F2!important;border:1px solid #DDD6D0!important;border-radius:8px!important;box-shadow:0 12px 30px #352F2B20!important}
[data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stButton"] button{min-height:2.3rem;font-size:.76rem;white-space:nowrap}

/* ============ polish: finale-only system header ============ */
body:has(.mj-finale-shell) .mj-system-header{background:linear-gradient(110deg,#3B1724,#6B3042 58%,#35131F);border-bottom:1px solid #C9A86A;box-shadow:inset 0 -8px 20px rgba(33,8,18,.28),inset 0 1px 0 rgba(255,225,205,.12)}
body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header){background:linear-gradient(110deg,#3B1724,#6B3042 58%,#35131F);border-bottom:1px solid #C9A86A;box-shadow:inset 0 -8px 20px rgba(33,8,18,.28),inset 0 1px 0 rgba(255,225,205,.12)}
body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header) .mj-system-header{background:transparent;border-bottom:0;box-shadow:none}
body:has(.mj-finale-shell) .mj-system-header__brand{color:#D9A6A8;letter-spacing:.06em;text-shadow:0 0 14px rgba(217,166,168,.18)}
body:has(.mj-finale-shell) .mj-system-status b{color:#FFF1E9}
body:has(.mj-finale-shell) .mj-system-status small{color:#E7C6BB}
body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stPopover"] button[data-testid="baseButton-secondary"],body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stButton"]>button{background:rgba(107,48,66,.72)!important;border:1px solid rgba(217,166,168,.7)!important;border-color:rgba(217,166,168,.7)!important;color:#F7D9C9!important}
body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stPopover"] button[data-testid="baseButton-secondary"] p{color:inherit!important}
body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stButton"]>button p{color:inherit!important}
body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stPopover"] button[data-testid="baseButton-secondary"]:hover,body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header) [data-testid="stButton"]>button:hover{background:rgba(124,58,77,.84)!important;border-color:#D9A6A8!important;color:#FFF1E9!important}
body:has(.mj-finale-shell) [data-testid="stPopoverBody"]{background:rgba(59,23,36,.98)!important;border:1px solid #C9A86A!important;border-color:#C9A86A!important;color:#F7D9C9!important}
body:has(.mj-finale-shell) [data-testid="stPopoverBody"] p,body:has(.mj-finale-shell) [data-testid="stPopoverBody"] label{color:#F7D9C9!important}

/* ============ polish: de-box（发丝线优先，每屏一个抬升主面板） ============ */
body [data-testid="stExpander"]{border:0;border-bottom:1px solid #E0D5C8;background:transparent}
body .mj-case-intake{border:0;background:transparent;padding:.25rem 0 .45rem}
body .mj-case-intake>header{border-left:2px solid var(--mj-polish-wine)}
body .mj-case-intake article{border:0;border-left:1px solid #E3D8CA;background:transparent}
body .mj-stage-card{border:0;border-bottom:1px solid #E0D5C7;background:linear-gradient(180deg,rgba(250,247,241,.6),rgba(240,233,224,.35))}
body .mj-stage-card.is-current{border:0;border-bottom:2px solid var(--mj-polish-wine);background:#FFF9F5}
body .mj-stage-card.is-complete{border:0;border-bottom:1px solid var(--mj-polish-gold);background:rgba(250,246,236,.7)}
body .mj-stage-workbench{border:0;border-left:3px solid var(--mj-polish-wine);box-shadow:0 12px 34px rgba(63,48,54,.07);min-height:15rem}
body .mj-stage-workbench__grid article{border:0;border-right:1px solid #E7DCCF;background:rgba(255,252,247,.5)}
body .mj-stage-workbench__grid article:last-child{border-right:0}
body .mj-stage-workbench__grid article.is-change{border:0;border-right:1px solid #E7DCCF;background:rgba(255,247,244,.75)}
body .pressure-hero{border:0;border-bottom:1px solid #E3D8CA}
body .pressure-check-card{border:0;border-top:1px solid #DFD3C6;background:rgba(250,247,241,.55)}
body .pressure-check-card.is-active{border:0;border-top:2px solid var(--mj-polish-wine);background:#FFF9F5}
body .pressure-candidate-meta{border:0;border-left:2px solid #E0D5C8;background:rgba(250,247,241,.6)}
body .pressure-candidate-meta.is-selected{border:0;border-left:2px solid var(--mj-polish-wine)}
body .pressure-holdout-row{border:0;border-bottom:1px solid #E7DCCF;background:transparent}
body .blind-review-stage{border:0;border-top:1px solid #E0D5C8;border-bottom:1px solid #E0D5C8;background:rgba(238,230,220,.55)}

/* ============ polish: 阶段内容软切换（局部重跑的重建闪烁 → 淡入上浮过渡） ============ */
@keyframes mjStageIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
body .pressure-original-stage,body .pressure-check-grid,body .pressure-check-detail,body .pressure-dialogue-title,body .pressure-holdout-grid,body .blind-review-stage,body .pressure-terminal,body .pressure-current{animation:mjStageIn .34s cubic-bezier(.16,1,.3,1) both}
body [data-testid="stCustomComponentV1"]{animation:mjStageIn .34s cubic-bezier(.16,1,.3,1) both}
body .pressure-stage-bar,body .pressure-candidate-meta-grid{animation:mjStageIn .28s cubic-bezier(.16,1,.3,1) both}
""".strip()
