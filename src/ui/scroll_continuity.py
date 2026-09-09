"""Keep the Streamlit main scroll position stable across stage fragment reruns."""

from __future__ import annotations

from html import escape


def build_scroll_continuity_html(token: str) -> str:
    """Build the same-origin iframe bridge mounted by each stage fragment."""
    safe_token = escape(token, quote=True)
    return (
        f'<script data-scroll-continuity-token="{safe_token}">\n'
        + """
(() => {
  const parentWindow = window.parent;
  if (!parentWindow || parentWindow === window) {
    console.error("[scroll-continuity] expected a same-origin parent page");
    return;
  }

  const controllerKey = "__MJ_SCROLL_CONTINUITY__";
  const existing = parentWindow[controllerKey];
  const pendingScrollTop = existing ? existing.pendingScrollTop : null;
  if (existing && existing.handlePointerDown) {
    parentWindow.document.removeEventListener(
      "pointerdown",
      existing.handlePointerDown,
      true,
    );
  }

  const navigationLabels = new Set(["下一步", "上一步", "下一阶段", "上一阶段"]);
  const locateMain = () => {
    const main = parentWindow.document.querySelector("section.main");
    if (!main) {
      console.error("[scroll-continuity] parent section.main is missing");
      return null;
    }
    return main;
  };
  const controller = {
    pendingScrollTop,

    restore() {
      if (this.pendingScrollTop === null) return;

      const scrollTop = this.pendingScrollTop;
      parentWindow.requestAnimationFrame(() => {
        parentWindow.requestAnimationFrame(() => {
          const main = locateMain();
          if (!main) return;

          const maximum = Math.max(0, main.scrollHeight - main.clientHeight);
          main.scrollTop = Math.min(scrollTop, maximum);
          this.pendingScrollTop = null;

          if (parentWindow.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            return;
          }

          const workbench = window.frameElement?.closest("[data-testid='stVerticalBlock']") || main;
          workbench.classList.remove("mj-scroll-continuity-enter");
          void workbench.offsetWidth;
          workbench.classList.add("mj-scroll-continuity-enter");
          parentWindow.setTimeout(() => {
            workbench.classList.remove("mj-scroll-continuity-enter");
          }, 200);
        });
      });
    },
  };

  if (!parentWindow.document.getElementById("mj-scroll-continuity-style")) {
    const style = parentWindow.document.createElement("style");
    style.id = "mj-scroll-continuity-style";
    style.textContent = "@keyframes mj-scroll-continuity-fade { from { opacity: .82; } to { opacity: 1; } } .mj-scroll-continuity-enter { animation: mj-scroll-continuity-fade 200ms ease-out; }";
    parentWindow.document.head.appendChild(style);
  }

  controller.handlePointerDown = (event) => {
    const button = event.target.closest("button");
    if (!button || !navigationLabels.has(button.innerText.trim())) return;

    const main = locateMain();
    if (!main) return;
    controller.pendingScrollTop = main.scrollTop;
  };
  parentWindow.document.addEventListener(
    "pointerdown",
    controller.handlePointerDown,
    true,
  );

  parentWindow[controllerKey] = controller;
  controller.restore();
})();
</script>
"""
    ).strip()


def render_scroll_continuity(token: str) -> None:
    """Mount the bridge without adding visible layout to a Streamlit fragment."""
    import streamlit.components.v1 as components

    components.html(build_scroll_continuity_html(token), height=0)
