#!/usr/bin/env python3
"""
Pemo.io Google Ads Keywords Puller

Uses the DataForSEO API to pull keywords that pemo.io is advertising
on Google Ads and outputs them in a formatted table.

Usage:
    python pemo_ads_keywords.py
    python pemo_ads_keywords.py --format csv
    python pemo_ads_keywords.py --format markdown
    python pemo_ads_keywords.py --output pemo_keywords.csv --format csv
"""

import argparse
import json
import sys

import requests


DATAFORSEO_LOGIN = "john@housesofgrowth.com"
DATAFORSEO_PASSWORD = "b1ff115a6d7ceb85"

API_URL = "https://api.dataforseo.com/v3/dataforseo_labs/google/domain_ads_keywords/live"
TARGET_DOMAIN = "pemo.io"


def fetch_ads_keywords(target: str, limit: int = 100) -> dict:
    """Fetch Google Ads keywords for a domain from DataForSEO."""
    payload = json.dumps([{
        "target": target,
        "language_name": "English",
        "location_code": 2840,
        "limit": limit,
        "order_by": ["keyword_data.keyword_info.search_volume,desc"],
    }])

    response = requests.post(
        API_URL,
        auth=(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD),
        headers={"Content-Type": "application/json"},
        data=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def parse_keywords(api_response: dict) -> list[dict]:
    """Extract keyword rows from the API response."""
    rows = []
    tasks = api_response.get("tasks", [])
    for task in tasks:
        if task.get("status_code") != 20000:
            print(f"  Warning: task status {task.get('status_code')}: {task.get('status_message')}")
            continue
        for result in task.get("result", []):
            for item in result.get("items", []):
                kw_data = item.get("keyword_data", {})
                kw_info = kw_data.get("keyword_info", {})
                ad_info = item.get("ad_info", {}) or {}
                rows.append({
                    "keyword": kw_data.get("keyword", ""),
                    "search_volume": kw_info.get("search_volume", 0),
                    "cpc": kw_info.get("cpc", 0.0),
                    "competition": kw_info.get("competition", 0.0),
                    "competition_level": kw_info.get("competition_level", ""),
                    "ad_position": item.get("ad_position", ""),
                    "ad_type": item.get("ad_type", ""),
                    "title": ad_info.get("title", ""),
                    "description": ad_info.get("description", ""),
                    "url": ad_info.get("url", ""),
                })
    return rows


def format_table(rows: list[dict]) -> str:
    """Format rows as a plain-text aligned table."""
    if not rows:
        return "No keywords found."

    headers = ["#", "Keyword", "Search Vol", "CPC ($)", "Competition", "Level", "Ad Position", "Ad Type"]
    col_data = []
    for i, r in enumerate(rows, 1):
        col_data.append([
            str(i),
            r["keyword"],
            f"{r['search_volume']:,}",
            f"{r['cpc']:.2f}",
            f"{r['competition']:.2f}",
            r["competition_level"],
            str(r["ad_position"]),
            r["ad_type"],
        ])

    col_widths = [len(h) for h in headers]
    for row in col_data:
        for j, cell in enumerate(row):
            col_widths[j] = max(col_widths[j], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    lines = [sep, header_line, sep]
    for row in col_data:
        line = "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def format_markdown(rows: list[dict]) -> str:
    """Format rows as a Markdown table."""
    if not rows:
        return "No keywords found."

    lines = [
        "| # | Keyword | Search Vol | CPC ($) | Competition | Level | Ad Position | Ad Type |",
        "|---|---------|-----------|---------|-------------|-------|-------------|---------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['keyword']} | {r['search_volume']:,} | "
            f"{r['cpc']:.2f} | {r['competition']:.2f} | "
            f"{r['competition_level']} | {r['ad_position']} | {r['ad_type']} |"
        )
    return "\n".join(lines)


def format_csv(rows: list[dict]) -> str:
    """Format rows as CSV."""
    if not rows:
        return "No keywords found."

    lines = ["keyword,search_volume,cpc,competition,competition_level,ad_position,ad_type,title,description,url"]
    for r in rows:
        title = r["title"].replace('"', '""')
        desc = r["description"].replace('"', '""')
        url = r["url"].replace('"', '""')
        lines.append(
            f'"{r["keyword"]}",{r["search_volume"]},{r["cpc"]:.2f},'
            f'{r["competition"]:.2f},"{r["competition_level"]}",'
            f'"{r["ad_position"]}","{r["ad_type"]}","{title}","{desc}","{url}"'
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pull Google Ads keywords for pemo.io via DataForSEO")
    parser.add_argument("--format", choices=["table", "markdown", "csv"], default="table",
                        help="Output format (default: table)")
    parser.add_argument("--output", default=None, help="Write output to file")
    parser.add_argument("--limit", type=int, default=100, help="Max keywords to pull (default: 100)")
    args = parser.parse_args()

    print(f"Fetching Google Ads keywords for {TARGET_DOMAIN}...")
    try:
        raw = fetch_ads_keywords(TARGET_DOMAIN, limit=args.limit)
    except requests.exceptions.RequestException as exc:
        sys.exit(f"Error calling DataForSEO API: {exc}")

    # Save raw response for debugging
    with open("pemo_keywords_raw.json", "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  Raw API response saved to pemo_keywords_raw.json")

    rows = parse_keywords(raw)
    print(f"  Found {len(rows)} keywords.\n")

    if args.format == "csv":
        output = format_csv(rows)
    elif args.format == "markdown":
        output = format_markdown(rows)
    else:
        output = format_table(rows)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Output saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
