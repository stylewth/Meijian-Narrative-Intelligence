"""目标品牌档案：把品牌词从 prompt 与 schema 中解耦，支持第二品牌迁移验证。

默认档案是梅见（第一品牌），保证官方链路与既有回放语义不变；自定义使用
链路在启动前调用 :func:`set_active_brand` 注入目标品牌（例如果立方）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrandProfile:
    brand_name: str
    industry: str
    target_audience: str
    research_question: str
    competitors: tuple[str, ...] = ()
    extended_alternatives: tuple[str, ...] = ()
    notes: str = ""

    def prompt_block(self) -> str:
        lines = [
            "# 当前目标品牌档案",
            f"- 品牌名称：{self.brand_name}",
            f"- 所属行业：{self.industry}",
            f"- 目标人群：{self.target_audience}",
            f"- 研究问题：{self.research_question}",
        ]
        if self.competitors:
            lines.append(f"- 主要竞品：{'、'.join(self.competitors)}")
        if self.extended_alternatives:
            lines.append(f"- 扩展替代选择：{'、'.join(self.extended_alternatives)}")
        if self.notes:
            lines.append(f"- 项目备注：{self.notes}")
        lines.append(
            "分析中的“目标品牌/该品牌”一律指上述品牌；竞品仅按输入语料中的"
            "事实处理，不得虚构竞品信息。"
        )
        return "\n".join(lines)


DEFAULT_BRAND = BrandProfile(
    brand_name="梅见",
    industry="低度酒/梅酒",
    target_audience="法定饮酒年龄的成年消费者",
    research_question=(
        "梅见能够依靠哪些真实资产，在什么具体场景中，为消费者建立竞品难以"
        "完整替换且可以长期建设的选择理由"
    ),
    competitors=("RIO", "MissBerry", "十七光年", "CHOYA"),
)

_active_brand: BrandProfile = DEFAULT_BRAND


def set_active_brand(profile: BrandProfile) -> None:
    global _active_brand
    _active_brand = profile


def get_active_brand() -> BrandProfile:
    return _active_brand


def active_brand_prompt_block() -> str:
    return _active_brand.prompt_block()
