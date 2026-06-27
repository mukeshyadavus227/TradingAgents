"""Plain-English Writer: turns the whole run into a layman-friendly briefing.

Every other report in the system is written for finance professionals. This
final stage reads the completed analyst reports, the research-team plan, the
trader's proposal, and the Portfolio Manager's decision, then synthesises them
into one professional briefing a non-expert can read and act on. It adds no new
analysis — it translates the firm's existing conclusions into plain English.

Follows the same structured-output pattern as the Portfolio Manager: the LLM
fills a typed ``PlainEnglishBriefing``, which is rendered back to markdown for
storage in ``plain_english_report`` so the CLI and saved reports consume a
single, consistent shape. When a provider lacks structured output, it falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import (
    PlainEnglishBriefing,
    render_plain_english_briefing,
)
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_plain_english_writer(llm):
    structured_llm = bind_structured(llm, PlainEnglishBriefing, "Plain English Writer")

    def plain_english_writer_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        def section(title: str, value: str) -> str:
            value = (value or "").strip()
            return f"## {title}\n{value}" if value else ""

        debate = state.get("investment_debate_state", {}) or {}
        reports = [
            section("Market analyst report", state.get("market_report", "")),
            section("Sentiment analyst report", state.get("sentiment_report", "")),
            section("News analyst report", state.get("news_report", "")),
            section("Fundamentals analyst report", state.get("fundamentals_report", "")),
            section("Research team plan", state.get("investment_plan", "") or debate.get("judge_decision", "")),
            section("Trader's proposal", state.get("trader_investment_plan", "")),
            section("Portfolio Manager's final decision", state.get("final_trade_decision", "")),
        ]
        source_material = "\n\n".join(part for part in reports if part)

        prompt = f"""You are the firm's editor. Your job is to translate the trading team's professional analysis into a single, plain-English briefing for a smart reader who has NO finance background.

{instrument_context}

---

Source material (the team's own reports — your only source of facts):

{source_material}

---

Write a professional briefing in plain, everyday English:
- Lead with the bottom line, then explain it.
- Avoid jargon. If you must use a technical term, define it in plain words the first time.
- Do NOT introduce any fact, number, or claim that is not supported by the source material above. Add no new analysis.
- Be honest about risk and uncertainty — do not oversell.
- Keep it concise and easy to scan.{get_language_instruction()}"""

        plain_english_report = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_plain_english_briefing,
            "Plain English Writer",
        )

        return {"plain_english_report": plain_english_report}

    return plain_english_writer_node
