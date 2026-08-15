"""集中管理梅见演示界面的 CSS 主题与动效边界。"""

__all__ = ["build_theme_css"]


def build_theme_css() -> str:
    """Return the single CSS payload injected by the UI shell."""

    return """
[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu {
  visibility: hidden;
}

[data-testid="stAppViewContainer"] {
  background: #F4EFE7;
  color: #211E1C;
}

[data-testid="stMainBlockContainer"] {
  width: min(100%, 94rem);
  max-width: 94rem;
  padding: 2rem clamp(1.25rem, 4vw, 4.5rem) 5rem;
}

.stApp,
.stApp button,
.stApp input,
.stApp textarea,
.stApp select {
  font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
}

.stApp h1,
.stApp h2,
.stApp h3 {
  font-family: STZhongsong, "华文中宋", "Source Han Serif SC", "Noto Serif SC", serif;
  letter-spacing: .02em;
}

.mj-workspace-nav {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 1rem;
  margin: 0 0 .8rem;
  padding: .5rem 0 .85rem;
  border-bottom: 1px solid #D8CCBE;
}

.mj-workspace-nav span {
  color: #8F2F4D;
  font-size: .7rem;
  font-weight: 750;
  letter-spacing: .16em;
}

.mj-workspace-nav strong {
  color: #4B4642;
  font-family: STZhongsong, "华文中宋", serif;
  font-size: .92rem;
  font-weight: 500;
  letter-spacing: .08em;
}

[class*="st-key-workspace-nav-"] button {
  min-height: 3.2rem;
  border: 1px solid #D7C9BC;
  border-radius: .35rem;
  background: #FAF7F1;
  box-shadow: none;
  color: #3B3734;
  font-size: .9rem;
}

[class*="st-key-workspace-nav-"] button:hover {
  border-color: #8F2F4D;
  color: #8F2F4D;
}

.mj-stage-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: .65rem;
  margin: 1.4rem 0 1rem;
}

.mj-stage-card {
  position: relative;
  display: grid;
  gap: .35rem;
  min-height: 8.6rem;
  padding: 1rem;
  overflow: hidden;
  border: 1px solid #D8CCBE;
  border-radius: .25rem;
  background: #FAF7F1;
}

.mj-stage-card .mj-stage-index {
  color: #A89C91;
  font-family: Georgia, serif;
  font-size: .72rem;
  letter-spacing: .12em;
}

.mj-stage-card strong {
  align-self: end;
  color: #403A36;
  font-family: STZhongsong, "华文中宋", serif;
  font-size: 1.15rem;
}

.mj-stage-card small {
  color: #877E77;
  font-size: .72rem;
}

.mj-stage-card i {
  position: absolute;
  inset: auto 0 0;
  height: 3px;
  background: #D8CCBE;
}

.mj-stage-card.is-complete {
  border-color: #8AA194;
  background: #F1F3EE;
}

.mj-stage-card.is-complete i {
  background: #496A5A;
}

.mj-stage-card.is-current {
  border-color: #8F2F4D;
  background: #FFF9F5;
  box-shadow: 0 .8rem 2rem rgba(63, 48, 54, .08);
  transform: translateY(-.2rem);
}

.mj-stage-card.is-current i {
  background: #8F2F4D;
}

.mj-stage-card.is-blocked {
  opacity: .6;
}

.mj-current-stage {
  display: grid;
  grid-template-columns: auto auto 1fr;
  gap: .75rem;
  align-items: baseline;
  margin: .2rem 0 1rem;
  padding: .9rem 1rem;
  border-left: 3px solid #8F2F4D;
  background: #ECE5DC;
}

.mj-current-stage span {
  color: #8F2F4D;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .12em;
}

.mj-current-stage p {
  margin: 0;
  color: #6C645E;
  font-size: .85rem;
}

.mj-stage-workbench {
  margin: .35rem 0 1.1rem;
  padding: 1.25rem;
  border: 1px solid #D7C9BC;
  border-left: 4px solid #8F2F4D;
  border-radius: .4rem;
  background: #F7F1E9;
  box-shadow: 0 1rem 2.5rem rgba(63, 48, 54, .07);
}

.mj-stage-workbench__header {
  display: flex;
  align-items: baseline;
  gap: .8rem;
  margin-bottom: 1rem;
}

.mj-stage-workbench__header span,
.mj-stage-workbench article small,
.mj-opportunity-card small {
  color: #8F2F4D;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .12em;
}

.mj-stage-workbench__header h2 {
  margin: 0;
  color: #3F3036;
  font-family: STZhongsong, "华文中宋", serif;
  font-size: 1.45rem;
}

.mj-stage-workbench__grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .75rem;
}

.mj-stage-workbench article {
  min-height: 8.8rem;
  padding: 1rem;
  border: 1px solid #DED3C8;
  background: rgba(255, 252, 247, .78);
}

.mj-stage-workbench article.is-change {
  border-color: rgba(143, 47, 77, .45);
  background: #FFF7F4;
}

.mj-stage-workbench article p {
  margin: .55rem 0 0;
  color: #4A433E;
  font-size: .82rem;
  line-height: 1.7;
}

.mj-stage-workbench article ul {
  display: flex;
  flex-wrap: wrap;
  gap: .4rem;
  margin: .75rem 0 0;
  padding: 0;
  list-style: none;
}

.mj-stage-workbench article li {
  display: grid;
  gap: .08rem;
  padding: .35rem .5rem;
  border: 1px solid #DFC9C8;
  background: #FFFDF9;
}

.mj-stage-workbench article li strong {
  color: #7C7068;
  font-size: .58rem;
}

.mj-stage-workbench article li span {
  color: #8F2F4D;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.mj-opportunity-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: .8rem;
  padding: .35rem 0;
}

.mj-opportunity-card {
  grid-column: span 2;
  display: grid;
  align-content: start;
  gap: .9rem;
  padding: 1.15rem;
  border: 1px solid #D8CCBE;
  border-top: 3px solid #8F2F4D;
  background: #FFFCF7;
}

.mj-opportunity-card:nth-child(-n+2) {
  grid-column: span 3;
}

.mj-opportunity-card header {
  display: flex;
  justify-content: space-between;
  color: #8F2F4D;
  font-size: .68rem;
  letter-spacing: .08em;
}

.mj-opportunity-card header strong {
  font-family: Georgia, serif;
  font-size: 1.35rem;
}

.mj-opportunity-card h3 {
  min-height: 3.1rem;
  margin: 0;
  color: #3F3036;
  font-family: STZhongsong, "华文中宋", serif;
  font-size: 1.05rem;
  line-height: 1.45;
}

.mj-opportunity-card section p {
  margin: .35rem 0 0;
  color: #5D544E;
  font-size: .76rem;
  line-height: 1.65;
}

.mj-opportunity-card footer {
  display: grid;
  gap: .3rem;
  padding-top: .75rem;
  border-top: 1px solid #E6DDD4;
  color: #81766E;
  font-size: .68rem;
  line-height: 1.5;
}

.mj-paper-shell,
.mj-live-shell,
.mj-finale-shell {
  --mj-color-ink: #0D1417;
  --mj-color-mint: #9FE0D1;
  --mj-color-plum: #3F3036;
  --mj-color-fluorescence: #9FE0D1;
  --mj-color-paper: #F5F0E8;
  --mj-color-paper-muted: #D9CEC0;
  --mj-color-focus: #F6C978;
  --mj-font-display: STZhongsong, "Noto Serif CJK SC", serif;
  --mj-font-body: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  --mj-motion-fast: 180ms;
  --mj-motion-slow: 720ms;
}

.mj-paper-shell {
  display: grid;
  gap: 1.5rem;
  max-width: 72rem;
  margin: 0 auto;
  padding: clamp(1.25rem, 4vw, 3rem);
  background-color: var(--mj-color-paper);
  color: var(--mj-color-ink);
  font-family: var(--mj-font-body);
  line-height: 1.65;
}

.mj-paper-title,
.mj-paper-heading {
  max-width: 44rem;
  margin: 0;
  color: var(--mj-color-plum);
  font-family: var(--mj-font-display);
  line-height: 1.2;
}

.mj-paper-panel,
.mj-paper-card {
  display: grid;
  gap: 0.75rem;
  padding: 1.25rem;
  border: 1px solid color-mix(in srgb, var(--mj-color-plum) 22%, transparent);
  background-color: color-mix(in srgb, var(--mj-color-paper) 88%, white);
}

.mj-paper-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  color: color-mix(in srgb, var(--mj-color-ink) 68%, var(--mj-color-paper));
  font-size: 0.875rem;
}

.mj-live-shell {
  display: grid;
  gap: 1rem;
  min-width: 0;
  padding: clamp(1rem, 3vw, 2rem);
  background: #0D1417;
  color: #FFF8EF;
  font-family: var(--mj-font-body);
  line-height: 1.5;
}

.mj-live-dashboard {
  display: grid;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid rgba(148, 163, 167, .18);
  border-radius: .5rem;
  background: #151F23;
  color: #ECE9E3;
}

.mj-live-kicker {
  color: #9FE0D1;
  font-size: .7rem;
  font-weight: 750;
  letter-spacing: .17em;
}

.mj-live-heading {
  margin: .15rem 0 .3rem;
  color: #ECE9E3;
  font-family: STZhongsong, "华文中宋", serif;
  font-size: clamp(2rem, 4vw, 3.8rem);
  font-weight: 500;
}

.mj-live-subtitle {
  max-width: 50rem;
  color: #94A3A7;
}

.mj-live-summary {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 1rem;
  align-items: center;
  padding: .8rem 1rem;
  border: 1px solid rgba(148, 163, 167, .18);
  background: #19262B;
}

.mj-live-summary b {
  color: #DFA0A3;
  font-family: Georgia, serif;
  font-size: 1.4rem;
}

.mj-opportunity-selector {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .7rem;
  margin: .8rem 0 .45rem;
}

.mj-opportunity-switch {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: .7rem;
  align-items: center;
  padding: .8rem 1rem;
  border: 1px solid rgba(148, 163, 167, .2);
  background: #151F23;
  color: #94A3A7;
}

.mj-opportunity-switch > span {
  font-family: Georgia, serif;
  font-size: .68rem;
}

.mj-opportunity-switch strong {
  color: #CDD4D3;
  font-size: .82rem;
}

.mj-opportunity-switch small {
  font-size: .64rem;
}

.mj-opportunity-switch.is-selected {
  border-color: #9FE0D1;
  background: #202F34;
  box-shadow: 0 0 1.25rem rgba(159, 224, 209, .1);
}

.mj-opportunity-switch.is-selected strong,
.mj-opportunity-switch.is-selected small {
  color: #9FE0D1;
}

.mj-live-phase {
  display: grid;
  grid-template-columns: auto auto 1fr;
  gap: .8rem;
  align-items: baseline;
  margin: 1rem 0;
  padding: .7rem 1rem;
  border-left: 3px solid #9FE0D1;
  background: #19262B;
}

.mj-live-phase > span {
  color: #9FE0D1;
  font-family: Georgia, serif;
  font-size: .7rem;
}

.mj-live-phase strong { color: #ECE9E3; }
.mj-live-phase p { margin: 0; color: #94A3A7; font-size: .75rem; }

.mj-blind-confirmation,
.mj-attribution-strip {
  margin: .8rem 0;
  padding: 1.2rem;
  border: 1px solid rgba(159, 224, 209, .22);
  background: #151F23;
}

.mj-blind-confirmation > span,
.mj-attribution-strip header span {
  color: #9FE0D1;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .12em;
}

.mj-blind-confirmation h3,
.mj-attribution-strip h3 {
  margin: .45rem 0;
  color: #ECE9E3;
  font-size: 1.1rem;
}

.mj-blind-confirmation p,
.mj-attribution-strip > p {
  margin: .4rem 0 0;
  color: #94A3A7;
  font-size: .75rem;
}

.mj-attribution-strip > div {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .65rem;
  margin-top: .8rem;
}

.mj-attribution-strip article {
  display: grid;
  gap: .3rem;
  padding: .8rem;
  background: #202F34;
}

.mj-attribution-strip article small { color: #94A3A7; }
.mj-attribution-strip article strong { color: #CDD4D3; font-size: .78rem; }

.mj-live-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(22rem, .85fr);
  gap: 1rem;
  align-items: start;
}

.mj-live-panel {
  min-width: 0;
  padding: 1rem;
  border: 1px solid rgba(148, 163, 167, .18);
  border-radius: .4rem;
  background: #19262B;
}

.mj-narrative-stage {
  position: relative;
  min-height: 13rem;
  padding: 1.2rem;
  overflow: hidden;
  border: 1px solid rgba(159, 224, 209, .28);
  background: #202F34;
}

.mj-narrative-stage > span {
  color: #9FE0D1;
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .12em;
}

.mj-narrative-stage h3 {
  margin: .55rem 0 1rem;
  color: #ECE9E3;
  font-size: 1.1rem;
}

.mj-narrative-stage p,
.mj-narrative-diff {
  margin: 0;
  color: #ECE9E3;
  font-family: STZhongsong, "华文中宋", serif;
  font-size: 1.06rem;
  line-height: 1.85;
}

.mj-narrative-stage.is-changing .mj-narrative-diff {
  animation: mj-narrativeFadeIn var(--mj-motion-slow) ease-out both;
}

.mj-narrative-old,
.mj-narrative-new,
.mj-narrative-static {
  color: #ECE9E3;
  font-family: STZhongsong, "华文中宋", serif;
  font-size: 1.06rem;
  line-height: 1.85;
}

.mj-narrative-old {
  animation: mj-narrative-old-out .65s ease .6s both;
}

.mj-narrative-new {
  margin-top: -2rem;
  transform: translateY(12px);
  animation: mj-narrative-new-in .8s ease 1.05s both;
}

.mj-narrative-added {
  padding: .05em .14em;
  background: rgba(159, 224, 209, .18);
  color: #BDF3E7;
}

.mj-narrative-removed {
  opacity: .34;
  text-decoration: line-through;
}

.mj-narrative-stage details {
  margin-top: 1rem;
  color: #94A3A7;
  font-size: .7rem;
}

.mj-narrative-stage details > div {
  margin-top: .55rem;
  padding: .65rem;
  background: #19262B;
}

.mj-narrative-stage details p { margin-top: .3rem; font-size: .74rem; }

@keyframes mj-narrative-old-out {
  0%, 35% { opacity: 1; transform: translateY(0); }
  100% { opacity: 0; transform: translateY(-10px); }
}

@keyframes mj-narrative-new-in {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

.mj-score-hero {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .55rem;
  margin: .7rem 0;
}

.mj-score-hero span {
  display: grid;
  gap: .2rem;
  padding: .75rem;
  background: #202F34;
}

.mj-score-hero small {
  color: #94A3A7;
  font-size: .68rem;
}

.mj-score-hero strong {
  color: #9FE0D1;
  font-family: Georgia, serif;
  font-size: 1.25rem;
}

.mj-score-summary {
  display: grid;
  align-content: center;
  min-height: 13rem;
  padding: 1.2rem;
  border: 1px solid rgba(159, 224, 209, .22);
  background: #202F34;
}

.mj-score-summary > span { color: #94A3A7; font-size: .68rem; }
.mj-score-summary > strong { margin: .4rem 0 .7rem; color: #9FE0D1; font-family: Georgia, serif; font-size: 3rem; }
.mj-score-summary > div { display: grid; grid-template-columns: 1fr 1fr; gap: .55rem; }
.mj-score-summary p { display: grid; gap: .2rem; margin: 0; padding: .65rem; background: #19262B; }
.mj-score-summary small { color: #94A3A7; }
.mj-score-summary b { color: #ECE9E3; }

.mj-detail-tab {
  min-height: 14rem;
  padding: 1rem;
  border: 1px solid rgba(148, 163, 167, .18);
  background: #19262B;
}

.mj-detail-tab h4 { margin: 0 0 .8rem; color: #ECE9E3; }
.mj-detail-tab ul { color: #CDD4D3; font-size: .74rem; line-height: 1.6; }
.mj-detail-tab li { margin: .45rem 0; }
.mj-detail-tab li span { display: block; color: #94A3A7; }
.mj-evidence-count { color: #94A3A7; }
.mj-evidence-count strong { color: #9FE0D1; font-size: 1.3rem; }

.mj-dimensions,
.mj-impact-grid section {
  padding: .9rem;
  background: #202F34;
}

.mj-dimensions h4,
.mj-impact-grid h4 {
  margin: 0 0 .75rem;
  color: #ECE9E3;
  font-size: .78rem;
  letter-spacing: .08em;
}

.mj-dimension-row {
  display: grid;
  grid-template-columns: 7.2rem minmax(4rem, 1fr) 2.2rem;
  gap: .55rem;
  align-items: center;
  margin: .5rem 0;
}

.mj-dimension-row span,
.mj-dimension-row strong {
  color: #CDD4D3;
  font-size: .72rem;
}

.mj-dimension-row i {
  height: 4px;
  overflow: hidden;
  background: #354449;
}

.mj-dimension-row i b {
  display: block;
  height: 100%;
  background: #9FE0D1;
}

.mj-dimension-row small {
  grid-column: 1 / -1;
  color: #94A3A7;
  font-size: .68rem;
  line-height: 1.45;
}

.mj-impact-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .7rem;
  margin-top: .7rem;
}

.mj-impact-grid ul {
  margin: 0;
  padding-left: 1rem;
  color: #B8C2C2;
  font-size: .72rem;
}

.mj-impact-grid li {
  margin: .35rem 0;
}

.mj-impact-grid li span {
  display: block;
  color: #94A3A7;
}

.mj-decision-brief {
  margin: .7rem 0 0;
  padding: .8rem;
  border-left: 2px solid #DFA0A3;
  color: #CDD4D3;
  font-size: .75rem;
  line-height: 1.6;
}

.mj-evidence-ticker {
  padding: .8rem 1rem;
  border: 1px solid rgba(159, 224, 209, .2);
  background: #151F23;
  color: #ECE9E3;
}

.mj-evidence-ticker h3 {
  margin: 0 0 .7rem;
  color: #9FE0D1;
  font-size: .72rem;
  letter-spacing: .12em;
}

.mj-evidence-ticker__item {
  padding: .55rem .8rem;
  border: 1px solid rgba(148, 163, 167, .18);
  background: #202F34;
  color: #CDD4D3;
}

.mj-evidence-ticker details {
  margin-top: .65rem;
  color: #94A3A7;
  font-size: .72rem;
}

.mj-comment-stage {
  min-height: 8.6rem;
}

.mj-comment-bullet strong { color: #9FE0D1; }
.mj-comment-bullet span { color: #CDD4D3; }

.mj-live-panel + div .js-line {
  stroke-dasharray: 1200;
  stroke-dashoffset: 1200;
  animation: mj-line-reveal .9s ease-out forwards;
}

@keyframes mj-line-reveal { to { stroke-dashoffset: 0; } }

.mj-live-line {
  display: grid;
  grid-template-columns: minmax(7rem, 0.3fr) minmax(0, 1fr);
  gap: 1rem;
  align-items: center;
  min-height: 3.5rem;
  padding: 0.75rem 1rem;
  border-inline-start: 2px solid color-mix(in srgb, var(--mj-color-mint) 30%, transparent);
  background-color: color-mix(in srgb, var(--mj-color-ink) 92%, var(--mj-color-plum));
}

.mj-live-line.is-active {
  border-inline-start-color: var(--mj-color-fluorescence);
  background-color: color-mix(in srgb, var(--mj-color-plum) 62%, var(--mj-color-ink));
  color: var(--mj-color-fluorescence);
  box-shadow: 0 0 1.25rem color-mix(in srgb, var(--mj-color-fluorescence) 26%, transparent);
}

.mj-live-node {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
  padding: 0.45rem 0.65rem;
  border: 1px solid color-mix(in srgb, var(--mj-color-mint) 36%, transparent);
  color: var(--mj-color-mint);
}

.mj-live-node.is-active {
  border-color: var(--mj-color-fluorescence);
  background-color: color-mix(in srgb, var(--mj-color-fluorescence) 12%, var(--mj-color-ink));
  color: var(--mj-color-fluorescence);
  box-shadow: 0 0 0.9rem color-mix(in srgb, var(--mj-color-fluorescence) 22%, transparent);
}

.mj-live-score {
  justify-self: end;
  min-width: 4.5rem;
  padding: 0.35rem 0.6rem;
  border: 1px solid color-mix(in srgb, var(--mj-color-mint) 40%, transparent);
  color: var(--mj-color-mint);
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.mj-live-score.is-active {
  border-color: var(--mj-color-fluorescence);
  color: var(--mj-color-fluorescence);
  box-shadow: 0 0 0.9rem color-mix(in srgb, var(--mj-color-fluorescence) 22%, transparent);
}

.mj-finale-shell {
  display: grid;
  place-items: center;
  gap: 1rem;
  min-height: 100vh;
  padding: clamp(2rem, 8vw, 6rem);
  overflow: hidden;
  background: #3E2C33;
  color: #FFF8EF;
  text-align: center;
}

.mj-finale-curtain {
  width: min(100%, 68rem);
  animation: mj-finalCurtainRise var(--mj-motion-slow) cubic-bezier(0.2, 0.7, 0.2, 1) both;
}

.mj-finale-slogan {
  display: grid;
  gap: 0.25rem;
  max-width: 48rem;
  margin: 0;
  color: #FFF8EF;
  font-family: var(--mj-font-display);
  line-height: 1.15;
}

.mj-finale-slogan-line-one,
.mj-finale-line-one {
  animation: mj-sloganLineOne var(--mj-motion-slow) ease-out 120ms both;
}

.mj-finale-slogan-line-two,
.mj-finale-line-two {
  animation: mj-sloganLineTwo var(--mj-motion-slow) ease-out 260ms both;
}

.mj-finale-kicker {
  display: block;
  margin-bottom: 1.2rem;
  color: #D5C295;
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .18em;
}

.mj-finale-line {
  display: block;
  width: fit-content;
  margin-inline: auto;
  font-size: clamp(2.2rem, 5.8vw, 5.8rem);
  letter-spacing: .06em;
  white-space: nowrap;
}

.mj-finale-line-two {
  font-size: clamp(2.45rem, 6.5vw, 6.5rem);
  letter-spacing: .09em;
}

.mj-hanging-mark {
  display: inline-block;
  width: 0;
  transform: translateX(.08em);
}

.mj-finale-slogan .mj-hanging-mark,
.mj-finale-scene-level {
  color: #D5C295;
}

.mj-finale-cta {
  display: inline-block;
  margin-top: 1.6rem;
  padding: .7rem 1.4rem;
  border: 1px solid rgba(197, 143, 162, .55);
  color: #C58FA2;
  font-size: .75rem;
  letter-spacing: .12em;
}

.mj-finale-act-two {
  animation: mj-finale-act-two-rise .8s cubic-bezier(.2,.75,.2,1) both;
}

.mj-finale-scenes,
.mj-finale-pillars,
.mj-finale-guardrails {
  display: grid;
  gap: 1rem;
  width: min(100%, 64rem);
  margin: 1.5rem auto 0;
  text-align: left;
}

.mj-finale-scenes {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.mj-finale-pillars {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.mj-finale-scene,
.mj-finale-pillar,
.mj-finale-guardrails section {
  padding: 1rem;
  border: 1px solid rgba(213, 194, 149, 0.34);
  background: rgba(98, 71, 81, 0.56);
}

.mj-finale-pillar {
  text-align: left;
}

.mj-finale-pillar > span {
  color: #D5C295;
  font-family: Georgia, serif;
  font-size: .72rem;
}

.mj-finale-pillar h3,
.mj-finale-pillar p {
  margin: .4rem 0 0;
}

.mj-finale-pillar p {
  color: #D9CEC0;
  font-size: .76rem;
  line-height: 1.65;
}

.mj-finale-scene h3,
.mj-finale-scene p,
.mj-finale-guardrails h3,
.mj-finale-guardrails ul {
  margin: 0.35rem 0 0;
}

.mj-finale-guardrails {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  color: #FFF8EF;
}

.mj-finale-narrative {
  display: -webkit-box;
  max-width: 44rem;
  margin: 1.3rem auto 0;
  overflow: hidden;
  color: #E4D5D2;
  line-height: 1.8;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
}

@keyframes mj-finale-act-two-rise {
  from { opacity: 0; transform: translateY(2rem); }
  to { opacity: 1; transform: translateY(0); }
}

.mj-finale-narrative.is-leaving {
  animation: mj-narrativeFadeOut var(--mj-motion-fast) ease-in both;
}

.mj-finale-narrative.is-entering {
  animation: mj-narrativeFadeIn var(--mj-motion-slow) ease-out both;
}

.mj-paper-shell :focus-visible,
.mj-live-shell :focus-visible,
.mj-finale-shell :focus-visible {
  outline: 2px solid var(--mj-color-focus);
  outline-offset: 3px;
}

/* Legacy animation names retained for compatibility: finalCurtainRise, sloganLineOne, sloganLineTwo, narrativeFadeOut, narrativeFadeIn. */
@keyframes mj-finalCurtainRise {
  from { opacity: 0; transform: translateY(2rem); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes mj-sloganLineOne {
  from { opacity: 0; transform: translateX(-1rem); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes mj-sloganLineTwo {
  from { opacity: 0; transform: translateX(1rem); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes mj-narrativeFadeOut {
  from { opacity: 1; }
  to { opacity: 0; }
}

@keyframes mj-narrativeFadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* Legacy aliases remain available for plan-era consumers; components use mj-* above. */
@keyframes finalCurtainRise {
  from { opacity: 0; transform: translateY(2rem); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes sloganLineOne {
  from { opacity: 0; transform: translateX(-1rem); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes sloganLineTwo {
  from { opacity: 0; transform: translateX(1rem); }
  to { opacity: 1; transform: translateX(0); }
}

@keyframes narrativeFadeOut {
  from { opacity: 1; }
  to { opacity: 0; }
}

@keyframes narrativeFadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@media (prefers-reduced-motion: reduce) {
  .mj-paper-shell *,
  .mj-live-shell *,
  .mj-finale-shell * {
    animation-duration: 0.01ms !important;
    animation-delay: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

@media (max-width: 760px) {
  .mj-stage-grid,
  .mj-live-grid,
  .mj-score-hero,
  .mj-impact-grid,
  .mj-finale-pillars,
  .mj-finale-scenes,
  .mj-finale-guardrails {
    grid-template-columns: 1fr;
  }

  .mj-current-stage {
    grid-template-columns: 1fr;
  }

  .mj-opportunity-selector,
  .mj-attribution-strip > div,
  .mj-live-phase {
    grid-template-columns: 1fr;
  }

  .mj-stage-workbench__grid,
  .mj-opportunity-grid {
    grid-template-columns: 1fr;
  }

  .mj-opportunity-card,
  .mj-opportunity-card:nth-child(-n+2) {
    grid-column: auto;
  }
}
""".strip()
