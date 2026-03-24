import os
import json
from typing import Any, Dict, List, Tuple

import streamlit as st
from openai import OpenAI

# ---------------------------------
# Config
# ---------------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are Operator OS, a high-leverage research and decision-support agent for founders, chiefs of staff, operators, and investors.

Your job is NOT to merely rewrite text.
Your job is to:
1. Break ambiguous business questions into researchable sub-questions.
2. Use web research when helpful.
3. Distinguish facts, assumptions, risks, and open questions.
4. Produce operator-grade recommendations, not generic summaries.
5. Cite sources when available.

Principles:
- Be practical, concise, and commercially sharp.
- State what you know, what you infer, and what remains uncertain.
- Prefer decision-useful output over exhaustive output.
- Surface tradeoffs clearly.
- Think like a chief of staff, strategy lead, or investor.
- Avoid fluff.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "research_question": {"type": "string"},
        "executive_summary": {"type": "string"},
        "sub_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "key_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "confidence": {"type": "string"}
                },
                "required": ["finding", "why_it_matters", "confidence"],
                "additionalProperties": False
            }
        },
        "market_landscape": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "details": {"type": "string"}
                },
                "required": ["theme", "details"],
                "additionalProperties": False
            }
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk": {"type": "string"},
                    "impact": {"type": "string"},
                    "mitigation": {"type": "string"}
                },
                "required": ["risk", "impact", "mitigation"],
                "additionalProperties": False
            }
        },
        "recommendation": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
                "rationale": {"type": "string"},
                "tradeoffs": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["verdict", "rationale", "tradeoffs"],
            "additionalProperties": False
        },
        "next_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "owner": {"type": "string"},
                    "timing": {"type": "string"}
                },
                "required": ["action", "owner", "timing"],
                "additionalProperties": False
            }
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "research_question",
        "executive_summary",
        "sub_questions",
        "key_findings",
        "market_landscape",
        "risks",
        "recommendation",
        "next_steps",
        "open_questions"
    ],
    "additionalProperties": False
}


def build_user_prompt(question: str, context: str, mode: str) -> str:
    return f"""
Mode: {mode}

Primary question:
{question}

Additional context:
{context if context.strip() else 'No additional context provided.'}

Instructions:
- Research the question using web search.
- Decompose the problem into useful sub-questions.
- Prioritize high-signal findings.
- Be explicit about uncertainty.
- Return a recommendation a founder, chief of staff, investor, or operator could act on immediately.
"""


def run_research(question: str, context: str, mode: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    user_prompt = build_user_prompt(question, context, mode)

    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        tools=[
            {"type": "web_search_preview"}
        ],
        tool_choice="auto",
        max_tool_calls=6,
        include=["web_search_call.action.sources"],
        text={
            "format": {
                "type": "json_schema",
                "name": "operator_os_research_output",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )

    result = json.loads(response.output_text)
    sources = extract_sources(response)
    return result, sources


def extract_sources(response: Any) -> List[Dict[str, str]]:
    gathered: List[Dict[str, str]] = []
    seen = set()

    output_items = getattr(response, "output", []) or []
    for item in output_items:
        item_type = getattr(item, "type", None)
        if item_type == "web_search_call":
            action = getattr(item, "action", None)
            if not action:
                continue
            sources = getattr(action, "sources", None) or []
            for src in sources:
                title = getattr(src, "title", None) or "Untitled source"
                url = getattr(src, "url", None) or ""
                if url and url not in seen:
                    seen.add(url)
                    gathered.append({"title": title, "url": url})

    return gathered


def render_findings(findings: List[Dict[str, str]]):
    for i, item in enumerate(findings, 1):
        with st.container(border=True):
            st.markdown(f"### {i}. {item.get('finding', 'Finding')}")
            st.write(f"**Why it matters:** {item.get('why_it_matters', '-')}")
            st.write(f"**Confidence:** {item.get('confidence', '-')}")


def render_landscape(items: List[Dict[str, str]]):
    for i, item in enumerate(items, 1):
        with st.container(border=True):
            st.markdown(f"### {i}. {item.get('theme', 'Theme')}")
            st.write(item.get('details', '-'))


def render_risks(items: List[Dict[str, str]]):
    for i, item in enumerate(items, 1):
        with st.container(border=True):
            st.markdown(f"### {i}. {item.get('risk', 'Risk')}")
            st.write(f"**Impact:** {item.get('impact', '-')}")
            st.write(f"**Mitigation:** {item.get('mitigation', '-')}")


def render_next_steps(items: List[Dict[str, str]]):
    for i, item in enumerate(items, 1):
        with st.container(border=True):
            st.markdown(f"### {i}. {item.get('action', 'Action')}")
            st.write(f"**Owner:** {item.get('owner', '-')}")
            st.write(f"**Timing:** {item.get('timing', '-')}")


def load_demo(mode: str) -> Tuple[str, str]:
    demos = {
        "Market entry": (
            "Should we expand into Chicago with a premium apartment cleaning service focused on renters paying $2,700-$3,800 per month?",
            "Target customer is busy professionals ages 25-40 living in 1-2 bedroom apartments. We want to understand pricing, demand, competition, and whether the unit economics could support a premium position."
        ),
        "Investment memo": (
            "Is a boutique AI consulting studio an attractive investment or acquisition target over the next 24 months?",
            "Focus on market demand, competition, service model defensibility, margin profile, and what would make a firm stand out beyond generic AI implementation work."
        ),
        "GTM research": (
            "Should a premium popcorn brand launch first through boutique gyms in Chicago?",
            "Assess customer fit, competitor brands, price positioning, partnership feasibility, and likely GTM risks."
        ),
        "Custom": ("", ""),
    }
    return demos.get(mode, ("", ""))


def main():
    st.set_page_config(page_title="Operator OS", layout="wide")
    st.title("Operator OS — Research + Decision Engine")
    st.caption("An AI agent that researches a business question, synthesizes evidence, and produces an operator-grade recommendation.")

    with st.sidebar:
        st.header("Demo modes")
        mode = st.selectbox(
            "Choose a mode",
            ["Market entry", "Investment memo", "GTM research", "Custom"],
            index=0,
        )
        st.markdown(
            """
            **How to demo this**
            1. Load a sample or enter your own question.
            2. Add context.
            3. Run the agent.
            4. Show the recommendation, risks, and cited sources.
            """
        )

    demo_question, demo_context = load_demo(mode)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        question = st.text_area(
            "Research question",
            value=demo_question,
            placeholder="Example: Should we enter X market? Is this company an attractive acquisition target? What's the best GTM path for this product?",
            height=120,
        )
    with col_b:
        if st.button("Load demo scenario"):
            st.session_state["question"] = demo_question
            st.session_state["context"] = demo_context
            st.rerun()

    if "question" in st.session_state:
        default_question = st.session_state["question"]
    else:
        default_question = demo_question

    if "context" in st.session_state:
        default_context = st.session_state["context"]
    else:
        default_context = demo_context

    question = st.text_area(
        "Question to investigate",
        value=default_question,
        placeholder="Enter the business question you want the agent to research.",
        height=100,
    )

    context = st.text_area(
        "Additional context",
        value=default_context,
        placeholder="Add goals, constraints, timeline, customer profile, or any other useful context.",
        height=180,
    )

    run = st.button("Run research agent", type="primary")

    if run:
        if not question.strip():
            st.error("Please enter a research question.")
            st.stop()

        with st.spinner("Researching and synthesizing..."):
            try:
                result, sources = run_research(question, context, mode)
            except Exception as e:
                st.exception(e)
                st.info("If the error mentions the web search tool type, check the current OpenAI docs and SDK version. The Responses API supports built-in web search, but tool naming can change across versions.")
                st.stop()

        st.subheader("Executive Summary")
        st.write(result.get("executive_summary", "-"))

        rec = result.get("recommendation", {})
        with st.container(border=True):
            st.subheader("Recommendation")
            st.write(f"**Verdict:** {rec.get('verdict', '-')}")
            st.write(f"**Rationale:** {rec.get('rationale', '-')}")
            tradeoffs = rec.get("tradeoffs", [])
            if tradeoffs:
                st.write("**Tradeoffs:**")
                for item in tradeoffs:
                    st.write(f"- {item}")

        st.subheader("Sub-questions")
        for item in result.get("sub_questions", []):
            st.write(f"- {item}")

        left, right = st.columns(2)

        with left:
            st.subheader("Key Findings")
            render_findings(result.get("key_findings", []))

            st.subheader("Market / Research Landscape")
            render_landscape(result.get("market_landscape", []))

        with right:
            st.subheader("Risks")
            render_risks(result.get("risks", []))

            st.subheader("Next Steps")
            render_next_steps(result.get("next_steps", []))

        st.subheader("Open Questions")
        for item in result.get("open_questions", []):
            st.write(f"- {item}")

        st.subheader("Sources")
        if sources:
            for src in sources:
                st.markdown(f"- [{src['title']}]({src['url']})")
        else:
            st.write("No sources were returned in the tool metadata.")

        st.subheader("Raw JSON")
        st.code(json.dumps(result, indent=2), language="json")


if __name__ == "__main__":
    main()
