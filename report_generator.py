#!/usr/bin/env python3
"""
AI Search Visibility Report Generator

Generates an "AI Search Visibility Report" for B2B SaaS companies,
showing how visible they are in AI search engines compared to competitors.

Usage:
    python report_generator.py "Asana" --url "https://asana.com"
    python report_generator.py "Linear"
"""

import argparse
import json
import os
import re
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import anthropic
import openai
import requests
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-4-20250514"
PERPLEXITY_MODEL = "sonar"
MAX_PROMPTS = 15
PARALLEL_WORKERS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_from_claude_response(message) -> str:
    """Extract only text content from a Claude API response, ignoring tool_use blocks."""
    parts = []
    for block in message.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts)


def parse_json_response(text: str) -> dict | list:
    """Parse JSON from Claude's response, stripping any markdown fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def scrape_homepage(url: str) -> str:
    """Scrape a homepage and return the visible text content (truncated)."""
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "AISearchVisibilityBot/1.0"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Truncate to ~3000 chars to fit in context
        return text[:3000]
    except Exception as exc:
        print(f"  Warning: could not scrape {url}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Stage 1: Company Research
# ---------------------------------------------------------------------------

def stage_company_research(
    company: str, url: str | None, anthropic_key: str
) -> dict:
    """Extract structured company data using Claude."""
    print("\n[Stage 1/5] Researching company...")

    homepage_context = ""
    if url:
        print(f"  Scraping {url}...")
        homepage_text = scrape_homepage(url)
        if homepage_text:
            homepage_context = f"\n\nHere is text scraped from their homepage ({url}):\n{homepage_text}"

    client = anthropic.Anthropic(api_key=anthropic_key)

    prompt = textwrap.dedent(f"""\
        I need structured data about the B2B SaaS company "{company}".
        {f'Their website is {url}.' if url else ''}
        {homepage_context}

        Return a JSON object (raw JSON, no markdown fences) with these fields:
        - "company_name": the company name
        - "description": a 1-2 sentence description of what they do
        - "category": the primary software category (e.g., "project management", "CRM", "email marketing")
        - "category_variations": a list of 2-4 alternative ways to describe their category (e.g., ["work management", "task management", "team collaboration"])
        - "key_features": a list of 3-5 key features or capabilities
        - "target_audience": who their primary customers are (e.g., "mid-market engineering teams")
    """)

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = extract_text_from_claude_response(message)
    company_data = parse_json_response(text)
    print(f"  Category: {company_data.get('category', 'unknown')}")
    print(f"  Target audience: {company_data.get('target_audience', 'unknown')}")
    return company_data


# ---------------------------------------------------------------------------
# Stage 2: Competitor Identification
# ---------------------------------------------------------------------------

def stage_competitor_identification(
    company: str, company_data: dict, anthropic_key: str
) -> list[dict]:
    """Use Claude with web search to find top 5 competitors."""
    print("\n[Stage 2/5] Identifying competitors...")

    client = anthropic.Anthropic(api_key=anthropic_key)

    category = company_data.get("category", "software")
    prompt = textwrap.dedent(f"""\
        Find the top 5 direct SaaS competitors to "{company}" in the {category} space.
        Search for "{company} competitors" and "{company} alternatives" to find them.

        Return a JSON array (raw JSON, no markdown fences) where each element has:
        - "name": the competitor company name
        - "url": their website URL

        Only include direct competitors that are SaaS products in a similar category.
        Return exactly 5 competitors.
    """)

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    text = extract_text_from_claude_response(message)
    competitors = parse_json_response(text)

    for c in competitors:
        print(f"  Found competitor: {c['name']} ({c.get('url', 'N/A')})")

    return competitors


# ---------------------------------------------------------------------------
# Stage 3: Prompt Generation
# ---------------------------------------------------------------------------

def stage_prompt_generation(
    company: str, company_data: dict, competitors: list[dict]
) -> list[dict]:
    """Generate buying-intent prompts for AI search analysis."""
    print("\n[Stage 3/5] Generating search prompts...")

    prompts = []

    # Type 1: Competitor alternative prompts
    all_brands = [company] + [c["name"] for c in competitors]
    alt_modifiers = ["alternatives", "competitors"]

    for brand in all_brands:
        for modifier in alt_modifiers:
            prompts.append({
                "text": f"What are the best {brand} {modifier}?",
                "type": "competitor",
                "brand_trigger": brand,
                "modifier": modifier,
            })

    # Type 2: Category prompts
    category = company_data.get("category", "software")
    variations = company_data.get("category_variations", [])
    all_categories = [category] + variations

    category_modifiers = ["software", "tools", "apps", "platforms", "solutions"]

    for cat in all_categories:
        for mod in category_modifiers:
            prompts.append({
                "text": f"What are the best {cat} {mod}?",
                "type": "category",
                "category": cat,
                "modifier": mod,
            })

    # Add natural-language recommendation prompts
    for cat in all_categories:
        prompts.append({
            "text": f"What {cat} software should I use for my business?",
            "type": "category",
            "category": cat,
            "modifier": "recommendation",
        })

    # Cap at MAX_PROMPTS
    if len(prompts) > MAX_PROMPTS:
        # Prioritize: keep a balanced mix of competitor and category prompts
        competitor_prompts = [p for p in prompts if p["type"] == "competitor"]
        category_prompts = [p for p in prompts if p["type"] == "category"]

        # Keep roughly half from each type
        half = MAX_PROMPTS // 2
        prompts = competitor_prompts[:half] + category_prompts[: MAX_PROMPTS - half]

    print(f"  Generated {len(prompts)} prompts ({sum(1 for p in prompts if p['type'] == 'competitor')} competitor, {sum(1 for p in prompts if p['type'] == 'category')} category)")
    for p in prompts:
        print(f"    - [{p['type']}] {p['text']}")

    return prompts


# ---------------------------------------------------------------------------
# Stage 4: AI Search Analysis
# ---------------------------------------------------------------------------

def analyze_brand_mentions(
    response_text: str,
    prospect: str,
    competitors: list[dict],
) -> dict:
    """Analyze which brands are mentioned in an AI search response."""
    all_brands = [prospect] + [c["name"] for c in competitors]
    text_lower = response_text.lower()
    midpoint = len(response_text) // 2

    mentions = {}
    for brand in all_brands:
        brand_lower = brand.lower()
        count = text_lower.count(brand_lower)
        if count > 0:
            first_pos = text_lower.index(brand_lower)
            position = "first_half" if first_pos < midpoint else "second_half"
        else:
            position = None

        mentions[brand] = {
            "mentioned": count > 0,
            "count": count,
            "position": position,
        }

    return mentions


def run_single_prompt(
    prompt: dict,
    prospect: str,
    competitors: list[dict],
    perplexity_key: str,
) -> dict:
    """Run a single prompt through Perplexity and analyze results."""
    client = openai.OpenAI(
        api_key=perplexity_key,
        base_url="https://api.perplexity.ai",
    )

    response = client.chat.completions.create(
        model=PERPLEXITY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that recommends software products. "
                    "Provide detailed, specific recommendations with product names."
                ),
            },
            {"role": "user", "content": prompt["text"]},
        ],
    )

    response_text = response.choices[0].message.content
    mentions = analyze_brand_mentions(response_text, prospect, competitors)

    return {
        "prompt": prompt,
        "response": response_text,
        "mentions": mentions,
    }


def stage_ai_search_analysis(
    prompts: list[dict],
    prospect: str,
    competitors: list[dict],
    perplexity_key: str,
) -> list[dict]:
    """Run all prompts through Perplexity AI in parallel."""
    print("\n[Stage 4/5] Running AI search analysis...")
    print(f"  Running {len(prompts)} prompts with {PARALLEL_WORKERS} workers...")

    results = []

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(
                run_single_prompt, prompt, prospect, competitors, perplexity_key
            ): prompt
            for prompt in prompts
        }

        for i, future in enumerate(as_completed(futures), 1):
            prompt = futures[future]
            try:
                result = future.result()
                results.append(result)
                prospect_mentioned = result["mentions"].get(prospect, {}).get(
                    "mentioned", False
                )
                status = "VISIBLE" if prospect_mentioned else "NOT VISIBLE"
                print(f"  [{i}/{len(prompts)}] {status} - {prompt['text']}")
            except Exception as exc:
                print(f"  [{i}/{len(prompts)}] ERROR - {prompt['text']}: {exc}")
                results.append({
                    "prompt": prompt,
                    "response": "",
                    "mentions": {},
                    "error": str(exc),
                })

    return results


# ---------------------------------------------------------------------------
# Stage 5: Report Generation
# ---------------------------------------------------------------------------

def calculate_visibility_scores(
    results: list[dict],
    prospect: str,
    competitors: list[dict],
) -> dict:
    """Calculate what % of prompts each brand appears in."""
    all_brands = [prospect] + [c["name"] for c in competitors]
    total = len(results)
    if total == 0:
        return {brand: 0.0 for brand in all_brands}

    scores = {}
    for brand in all_brands:
        appearances = sum(
            1
            for r in results
            if r.get("mentions", {}).get(brand, {}).get("mentioned", False)
        )
        scores[brand] = round((appearances / total) * 100, 1)

    return scores


def generate_bar(score: float, max_width: int = 30) -> str:
    """Generate a text-based bar for the leaderboard."""
    filled = round(score / 100 * max_width)
    return "\u2588" * filled + "\u2591" * (max_width - filled)


def stage_report_generation(
    company: str,
    company_data: dict,
    competitors: list[dict],
    prompts: list[dict],
    results: list[dict],
) -> str:
    """Generate the final branded Markdown report."""
    print("\n[Stage 5/5] Generating report...")

    scores = calculate_visibility_scores(results, company, competitors)
    prospect_score = scores.get(company, 0.0)
    all_brands = [company] + [c["name"] for c in competitors]

    # Sort by visibility score descending
    leaderboard = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Find top competitor
    competitor_scores = {k: v for k, v in scores.items() if k != company}
    top_competitor = max(competitor_scores, key=competitor_scores.get) if competitor_scores else "N/A"
    top_competitor_score = competitor_scores.get(top_competitor, 0.0)

    # Separate results by type
    competitor_results = [r for r in results if r["prompt"]["type"] == "competitor"]
    category_results = [r for r in results if r["prompt"]["type"] == "category"]

    # Count missed opportunities
    missed = [
        r
        for r in results
        if not r.get("mentions", {}).get(company, {}).get("mentioned", False)
        and any(
            r.get("mentions", {}).get(c["name"], {}).get("mentioned", False)
            for c in competitors
        )
    ]

    total_prompts = len(results)
    prompts_visible = sum(
        1
        for r in results
        if r.get("mentions", {}).get(company, {}).get("mentioned", False)
    )
    prompts_missing = total_prompts - prompts_visible

    date_str = datetime.now().strftime("%B %d, %Y")

    # Build the report
    lines = []

    # Header
    lines.append(f"# AI Search Visibility Report: {company}")
    lines.append(f"*Generated on {date_str} by House of Growth*")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        f"**{company}** has an overall AI search visibility score of **{prospect_score}%** "
        f"across {total_prompts} high-intent buying prompts tested."
    )
    lines.append("")

    if prospect_score == 0:
        lines.append(
            f"{company} is currently **invisible** in AI search results. "
            f"When potential buyers ask AI assistants about {company_data.get('category', 'software')} "
            f"solutions, {company} does not appear in any of the responses we tested."
        )
    elif prospect_score < 30:
        lines.append(
            f"{company} has **low visibility** in AI search. "
            f"The company appears in only {prompts_visible} out of {total_prompts} tested prompts, "
            f"meaning potential buyers are unlikely to discover {company} through AI search."
        )
    elif prospect_score < 60:
        lines.append(
            f"{company} has **moderate visibility** in AI search. "
            f"The company appears in {prompts_visible} out of {total_prompts} tested prompts, "
            f"but there are significant gaps where competitors are visible and {company} is not."
        )
    else:
        lines.append(
            f"{company} has **strong visibility** in AI search. "
            f"The company appears in {prompts_visible} out of {total_prompts} tested prompts. "
            f"There may still be opportunities to close remaining gaps."
        )

    lines.append("")
    if top_competitor_score > prospect_score:
        gap = round(top_competitor_score - prospect_score, 1)
        lines.append(
            f"The most visible competitor, **{top_competitor}**, scores **{top_competitor_score}%** "
            f"-- **{gap} percentage points ahead** of {company}."
        )
    elif prospect_score > 0:
        lines.append(
            f"{company} is the **most visible** brand in this competitive set, "
            f"outperforming {top_competitor} ({top_competitor_score}%)."
        )
    lines.append("")

    # Visibility Leaderboard
    lines.append("## Visibility Leaderboard")
    lines.append("")
    lines.append("How often each brand appears across all tested AI search prompts:")
    lines.append("")
    lines.append("```")
    for brand, score in leaderboard:
        bar = generate_bar(score)
        marker = " <-- YOU" if brand == company else ""
        lines.append(f"  {brand:<20} {bar} {score:>5.1f}%{marker}")
    lines.append("```")
    lines.append("")

    # Competitor Prompt Breakdown
    if competitor_results:
        lines.append("## Competitor Alternative Prompts")
        lines.append("")
        lines.append(
            "These prompts test whether AI search engines recommend your brand "
            "when users search for alternatives to your competitors (and vice versa)."
        )
        lines.append("")

        for r in competitor_results:
            prompt_text = r["prompt"]["text"]
            prospect_visible = r.get("mentions", {}).get(company, {}).get("mentioned", False)

            lines.append(f"### `{prompt_text}`")
            lines.append("")
            lines.append(
                f"**{company}**: {'Mentioned' if prospect_visible else 'NOT mentioned'}"
            )

            # Show which competitors appeared
            visible_competitors = [
                c["name"]
                for c in competitors
                if r.get("mentions", {}).get(c["name"], {}).get("mentioned", False)
            ]
            if visible_competitors:
                lines.append(f"**Competitors visible**: {', '.join(visible_competitors)}")
            else:
                lines.append("**Competitors visible**: None")
            lines.append("")

    # Category Prompt Breakdown
    if category_results:
        lines.append("## Category Search Prompts")
        lines.append("")
        lines.append(
            "These prompts test whether AI search engines recommend your brand "
            "when users search for your software category."
        )
        lines.append("")

        for r in category_results:
            prompt_text = r["prompt"]["text"]
            prospect_visible = r.get("mentions", {}).get(company, {}).get("mentioned", False)

            lines.append(f"### `{prompt_text}`")
            lines.append("")
            lines.append(
                f"**{company}**: {'Mentioned' if prospect_visible else 'NOT mentioned'}"
            )

            visible_competitors = [
                c["name"]
                for c in competitors
                if r.get("mentions", {}).get(c["name"], {}).get("mentioned", False)
            ]
            if visible_competitors:
                lines.append(f"**Competitors visible**: {', '.join(visible_competitors)}")
            else:
                lines.append("**Competitors visible**: None")
            lines.append("")

    # Biggest Missed Opportunities
    if missed:
        lines.append("## Biggest Missed Opportunities")
        lines.append("")
        lines.append(
            "These are high-intent prompts where **competitors appear but you don't**. "
            "Each one represents potential buyers who are discovering your competitors instead of you."
        )
        lines.append("")

        for r in missed:
            prompt_text = r["prompt"]["text"]
            visible_competitors = [
                c["name"]
                for c in competitors
                if r.get("mentions", {}).get(c["name"], {}).get("mentioned", False)
            ]
            lines.append(
                f"- **\"{prompt_text}\"** -- Competitors visible: {', '.join(visible_competitors)}"
            )

        lines.append("")

    # Revenue Impact
    lines.append("## Revenue Impact")
    lines.append("")

    if prompts_missing > 0:
        lines.append(
            f"{company} is **missing from {prompts_missing} out of {total_prompts}** "
            f"high-intent buying prompts tested."
        )
        lines.append("")
        lines.append(
            "As AI search engines (ChatGPT, Perplexity, Claude) become a primary way "
            "B2B buyers discover and evaluate software, every prompt where you're invisible "
            "is a lost opportunity. Buyers who don't see you in AI results won't add you to "
            "their shortlist."
        )
    else:
        lines.append(
            f"{company} appears in all {total_prompts} tested prompts -- strong performance. "
            "However, visibility is just the first step. Position, sentiment, and recommendation "
            "strength also matter."
        )

    lines.append("")

    if top_competitor_score > prospect_score:
        lines.append(
            f"**{top_competitor}** is currently **{round(top_competitor_score - prospect_score, 1)} "
            f"percentage points more visible** than {company} in AI search. "
            f"This means {top_competitor} is being recommended to potential buyers "
            f"in searches where {company} is not."
        )
        lines.append("")

    # CTA
    lines.append("---")
    lines.append("")
    lines.append("## Want to Fix This?")
    lines.append("")
    lines.append(
        "**House of Growth** specializes in AI Search Optimization for B2B SaaS companies. "
        "We help you get recommended by AI search engines like ChatGPT, Perplexity, and Claude "
        "when your ideal buyers are searching for solutions like yours."
    )
    lines.append("")
    lines.append("**What we can do for you:**")
    lines.append("- Audit your full AI search presence across all major AI platforms")
    lines.append("- Identify and close visibility gaps against your competitors")
    lines.append("- Build an ongoing AI search optimization strategy")
    lines.append("- Track your AI search visibility over time")
    lines.append("")
    lines.append("**Book a free strategy call:** [houseofgrowth.com](https://houseofgrowth.com)")
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by House of Growth AI Search Visibility Tool | {date_str}*")
    lines.append("")

    report = "\n".join(lines)
    print(f"  Report generated: {len(report):,} characters")
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate an AI Search Visibility Report for a B2B SaaS company."
    )
    parser.add_argument("company", help="Company name (e.g. 'Asana', 'Linear')")
    parser.add_argument(
        "--url",
        default=None,
        help="Company website URL (optional, improves research accuracy)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: report_<company>_<date>.md)",
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
    date_slug = datetime.now().strftime("%Y%m%d")
    company_slug = company.lower().replace(" ", "_")
    output_path = args.output or f"report_{company_slug}_{date_slug}.md"

    print(f"\n{'=' * 60}")
    print(f"  AI Search Visibility Report Generator")
    print(f"  Company : {company}")
    print(f"  URL     : {url or '(not provided)'}")
    print(f"  Output  : {output_path}")
    print(f"{'=' * 60}")

    # Stage 1: Company Research
    company_data = stage_company_research(company, url, anthropic_key)

    # Stage 2: Competitor Identification
    competitors = stage_competitor_identification(company, company_data, anthropic_key)

    # Stage 3: Prompt Generation
    prompts = stage_prompt_generation(company, company_data, competitors)

    # Stage 4: AI Search Analysis
    results = stage_ai_search_analysis(prompts, company, competitors, perplexity_key)

    # Stage 5: Report Generation
    report = stage_report_generation(
        company, company_data, competitors, prompts, results
    )

    # Write output
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"  Done! Report saved to: {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
