import streamlit.components.v1 as components

from src.ui import scroll_continuity
from src.ui.scroll_continuity import build_scroll_continuity_html


def test_bridge_targets_only_stage_navigation_buttons():
    html = build_scroll_continuity_html("test")

    for label in ("下一步", "上一步", "下一阶段", "上一阶段"):
        assert label in html
    for label in ("重播", "揭晓 5→3", "进入实时决策看板"):
        assert label not in html


def test_bridge_restores_once_after_layout_and_honors_reduced_motion():
    html = build_scroll_continuity_html("test")

    assert "__MJ_SCROLL_CONTINUITY__" in html
    assert html.count("addEventListener") == 1
    assert html.count("requestAnimationFrame") >= 2
    assert "Math.min" in html
    assert "prefers-reduced-motion: reduce" in html


def test_bridge_rebinds_the_single_listener_after_fragment_replacement():
    html = build_scroll_continuity_html("test")

    assert "existing.pendingScrollTop" in html
    assert html.count("removeEventListener") == 1
    assert html.count("addEventListener") == 1
    assert html.index("removeEventListener") < html.index("addEventListener")


def test_bridge_animates_only_the_current_fragment_root():
    html = build_scroll_continuity_html("test")

    assert "window.frameElement" in html
    assert "stMainBlockContainer" not in html


def test_bridge_schedules_both_restore_frames_from_parent_page():
    html = build_scroll_continuity_html("test")

    assert html.count("parentWindow.requestAnimationFrame") == 2


def test_bridge_srcdoc_changes_when_the_safe_mount_token_changes():
    first = build_scroll_continuity_html("pressure:2")
    second = build_scroll_continuity_html("pressure:3")
    escaped = build_scroll_continuity_html('pressure:<3&"')

    assert first != second
    assert 'data-scroll-continuity-token="pressure:2"' in first
    assert 'data-scroll-continuity-token="pressure:3"' in second
    assert 'data-scroll-continuity-token="pressure:&lt;3&amp;&quot;"' in escaped
    assert 'pressure:<3&"' not in escaped


def test_renderer_builds_component_html_from_its_mount_token(monkeypatch):
    rendered: list[tuple[str, int]] = []
    monkeypatch.setattr(
        scroll_continuity,
        "build_scroll_continuity_html",
        lambda token: f"bridge:{token}",
    )
    monkeypatch.setattr(
        components,
        "html",
        lambda html, *, height: rendered.append((html, height)),
    )

    scroll_continuity.render_scroll_continuity("preprocessing:4")

    assert rendered == [("bridge:preprocessing:4", 0)]
