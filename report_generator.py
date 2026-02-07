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
from urllib.parse import urlparse

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

DATAFORSEO_API_URL = (
    "https://api.dataforseo.com/v3/dataforseo_labs/google/ranked_keywords/live"
)


# ---------------------------------------------------------------------------
# DataForSEO: paid keywords lookup
# ---------------------------------------------------------------------------

def fetch_paid_keywords(
    domain: str,
    login: str,
    password: str,
    location_code: int = 2840,
    language_name: str = "English",
    limit: int = 1000,
) -> dict:
    """Fetch paid (PPC) keywords for *domain* via DataForSEO Labs API.

    Returns a dict with:
        total_count  – number of paid keywords found
        keywords     – list of dicts with keyword details
    """
    payload = [
        {
            "target": domain,
            "language_name": language_name,
            "location_code": location_code,
            "filters": [
                ["ranked_serp_element.serp_item.type", "=", "paid"]
            ],
            "limit": limit,
        }
    ]

    resp = requests.post(
        DATAFORSEO_API_URL,
        json=payload,
        auth=(login, password),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    # Validate top-level status
    if data.get("status_code") != 20000:
        raise RuntimeError(
            f"DataForSEO request failed: {data.get('status_message', 'unknown error')}"
        )

    task = data["tasks"][0]
    if task.get("status_code") != 20000:
        raise RuntimeError(
            f"DataForSEO task error: {task.get('status_message', 'unknown error')}"
        )

    result = task.get("result", [{}])[0] if task.get("result") else {}
    total_count = result.get("total_count", 0)
    items = result.get("items") or []

    keywords = []
    for item in items:
        kw_data = item.get("keyword_data", {})
        kw_info = kw_data.get("keyword_info", {})
        serp_elem = item.get("ranked_serp_element", {})
        serp_item = serp_elem.get("serp_item", {})

        keywords.append({
            "keyword": kw_data.get("keyword", ""),
            "search_volume": kw_info.get("search_volume"),
            "cpc": kw_info.get("cpc"),
            "competition": kw_info.get("competition"),
            "position": serp_item.get("rank_group"),
            "title": serp_item.get("title", ""),
            "url": serp_item.get("url", ""),
        })

    return {"total_count": total_count, "keywords": keywords}


def print_paid_keywords_report(domain: str, result: dict) -> None:
    """Pretty-print paid keywords data to the console."""
    total = result["total_count"]
    keywords = result["keywords"]

    print(f"\n{'='*60}")
    print(f"  Paid Keywords Report for: {domain}")
    print(f"{'='*60}")
    print(f"\n  Total paid keywords: {total}\n")

    if not keywords:
        print("  No paid keywords found.")
        return

    # Header
    print(f"  {'#':<4} {'Keyword':<40} {'Vol':>8} {'CPC':>8} {'Pos':>5}")
    print(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*8} {'-'*5}")

    for i, kw in enumerate(keywords, 1):
        keyword = kw["keyword"][:40]
        vol = kw["search_volume"] if kw["search_volume"] is not None else "N/A"
        cpc = f"${kw['cpc']:.2f}" if kw["cpc"] is not None else "N/A"
        pos = kw["position"] if kw["position"] is not None else "N/A"
        print(f"  {i:<4} {keyword:<40} {vol:>8} {cpc:>8} {pos:>5}")

    print()


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
    7. Paid Search (PPC) Strategy (include only if paid-keyword data is provided)
    8. Key Takeaways (bullet points)

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
    parser.add_argument(
        "--paid-keywords",
        action="store_true",
        default=False,
        help="Fetch paid (PPC) keywords via DataForSEO and include in the report",
    )
    args = parser.parse_args()

    # Resolve API keys
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    perplexity_key = os.environ.get("PERPLEXITY_API_KEY")

    if not anthropic_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")
    if not perplexity_key:
        sys.exit("Error: PERPLEXITY_API_KEY environment variable is not set.")

    # DataForSEO credentials (required only when --paid-keywords is used)
    dataforseo_login = os.environ.get("DATAFORSEO_LOGIN")
    dataforseo_password = os.environ.get("DATAFORSEO_PASSWORD")

    if args.paid_keywords and (not dataforseo_login or not dataforseo_password):
        sys.exit(
            "Error: DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD environment "
            "variables are required when using --paid-keywords."
        )

    company = args.company
    url = args.url
    output_path = args.output or f"{company.lower().replace(' ', '_')}_report.md"

    # Extract bare domain from URL for DataForSEO queries
    domain = urlparse(url).netloc or url.replace("https://", "").replace("http://", "").split("/")[0]

    total_steps = 4 if args.paid_keywords else 3
    step = 0

    print(f"\n{'='*60}")
    print(f"  Lead Magnet Report Generator")
    print(f"  Company : {company}")
    print(f"  URL     : {url}")
    print(f"{'='*60}\n")

    # Step 1: Scrape metadata
    step += 1
    print(f"[{step}/{total_steps}] Scraping page metadata...")
    meta = scrape_meta(url)

    # Step 2 (optional): Fetch paid keywords via DataForSEO
    paid_kw_result = None
    if args.paid_keywords:
        step += 1
        print(f"[{step}/{total_steps}] Fetching paid keywords from DataForSEO...")
        try:
            paid_kw_result = fetch_paid_keywords(domain, dataforseo_login, dataforseo_password)
            print_paid_keywords_report(domain, paid_kw_result)
        except Exception as exc:
            print(f"  Warning: could not fetch paid keywords: {exc}")

    # Step N-1: Research via Perplexity
    step += 1
    print(f"[{step}/{total_steps}] Researching company via Perplexity AI...")
    research = research_company(company, url, perplexity_key)

    # Append paid keywords as extra research context if available
    if paid_kw_result and paid_kw_result["keywords"]:
        kw_summary_lines = [
            f"Total paid keywords: {paid_kw_result['total_count']}",
            "",
            "Top paid keywords:",
        ]
        for kw in paid_kw_result["keywords"][:50]:
            vol = kw["search_volume"] if kw["search_volume"] is not None else "N/A"
            cpc = f"${kw['cpc']:.2f}" if kw["cpc"] is not None else "N/A"
            kw_summary_lines.append(
                f"- {kw['keyword']} (volume: {vol}, CPC: {cpc})"
            )
        research.append({
            "question": f"What paid search (PPC) keywords is {company} bidding on?",
            "answer": "\n".join(kw_summary_lines),
        })

    # Step N: Generate report via Claude
    step += 1
    print(f"[{step}/{total_steps}] Generating report with Claude...")
    report = generate_report(company, url, research, meta, anthropic_key)

    # Write output
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nDone! Report saved to: {output_path}")
    print(f"  Length: {len(report):,} characters")


if __name__ == "__main__":
    main()
