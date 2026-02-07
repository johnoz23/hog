#!/usr/bin/env python3
"""
Lead Magnet Report Generator

Researches a company using Perplexity AI, then uses Claude to produce
a polished market-intelligence report suitable as a lead magnet.

Usage:
    python report_generator.py "CompanyName" --url "https://company.com"
"""

import argparse
import os
import sys
import json
import textwrap
from datetime import datetime

import anthropic
import openai
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERPLEXITY_MODEL = "sonar-pro"
CLAUDE_MODEL = "claude-sonnet-4-20250514"

RESEARCH_QUERIES = [
    "What does {company} ({url}) do? Describe the product, target market, and value proposition.",
    "Who are {company}'s main competitors and how does {company} differentiate?",
    "What recent news, funding rounds, or product launches has {company} had in the last 12 months?",
    "What are the biggest challenges and opportunities in {company}'s market segment?",
]


# ---------------------------------------------------------------------------
# Perplexity research layer
# ---------------------------------------------------------------------------

def perplexity_search(query: str, api_key: str) -> str:
    """Call the Perplexity chat completions endpoint and return the answer."""
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.perplexity.ai",
    )
    response = client.chat.completions.create(
        model=PERPLEXITY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior market-research analyst. "
                    "Provide detailed, factual answers with specifics "
                    "(numbers, dates, names) when available."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


def research_company(company: str, url: str, api_key: str) -> list[dict]:
    """Run all research queries and return a list of {question, answer} dicts."""
    results = []
    for template in RESEARCH_QUERIES:
        query = template.format(company=company, url=url)
        print(f"  Researching: {query[:80]}...")
        answer = perplexity_search(query, api_key)
        results.append({"question": query, "answer": answer})
    return results


# ---------------------------------------------------------------------------
# Optional: scrape basic page metadata
# ---------------------------------------------------------------------------

def scrape_meta(url: str) -> dict:
    """Fetch the target URL and pull basic metadata."""
    meta = {"title": "", "description": ""}
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "LeadMagnetBot/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        meta["title"] = soup.title.string.strip() if soup.title and soup.title.string else ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            meta["description"] = desc_tag["content"].strip()
    except Exception as exc:
        print(f"  Warning: could not scrape {url}: {exc}")
    return meta


# ---------------------------------------------------------------------------
# Claude synthesis layer
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert business analyst who writes compelling, data-rich
    market-intelligence reports.  Your reports are used as lead magnets —
    they must be insightful enough that a reader would willingly exchange
    their email to receive one.

    Structure every report with:
    1. Executive Summary (2-3 paragraphs)
    2. Company Overview
    3. Product & Value Proposition
    4. Competitive Landscape
    5. Recent Developments
    6. Market Opportunities & Challenges
    7. Key Takeaways (bullet points)

    Use markdown formatting. Be specific — cite numbers, dates, and names
    whenever the research supports it.  Avoid filler and generic statements.
""")


def generate_report(
    company: str,
    url: str,
    research: list[dict],
    meta: dict,
    api_key: str,
) -> str:
    """Send research to Claude and get back a finished report."""
    research_block = "\n\n".join(
        f"### Research: {r['question']}\n{r['answer']}" for r in research
    )

    user_prompt = textwrap.dedent(f"""\
        Write a comprehensive market-intelligence report for **{company}** ({url}).

        Page metadata:
        - Title: {meta.get('title', 'N/A')}
        - Description: {meta.get('description', 'N/A')}

        Below is the raw research collected from multiple queries.
        Synthesize it into a single cohesive report.

        ---
        {research_block}
        ---
    """)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=REPORT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a lead-magnet report for a company."
    )
    parser.add_argument("company", help="Company name (e.g. 'Linear')")
    parser.add_argument("--url", required=True, help="Company website URL")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: <company>_report.md)",
    )
    args = parser.parse_args()

    # Resolve API keys
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    perplexity_key = os.environ.get("PERPLEXITY_API_KEY")

    if not anthropic_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")
    if not perplexity_key:
        sys.exit("Error: PERPLEXITY_API_KEY environment variable is not set.")

    company = args.company
    url = args.url
    output_path = args.output or f"{company.lower().replace(' ', '_')}_report.md"

    print(f"\n{'='*60}")
    print(f"  Lead Magnet Report Generator")
    print(f"  Company : {company}")
    print(f"  URL     : {url}")
    print(f"{'='*60}\n")

    # Step 1: Scrape metadata
    print("[1/3] Scraping page metadata...")
    meta = scrape_meta(url)

    # Step 2: Research via Perplexity
    print("[2/3] Researching company via Perplexity AI...")
    research = research_company(company, url, perplexity_key)

    # Step 3: Generate report via Claude
    print("[3/3] Generating report with Claude...")
    report = generate_report(company, url, research, meta, anthropic_key)

    # Write output
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nDone! Report saved to: {output_path}")
    print(f"  Length: {len(report):,} characters")


if __name__ == "__main__":
    main()
