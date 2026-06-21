#!/usr/bin/env python3
"""Selenium scraper for AMC 10 problems from AoPS wiki.

Strategy: one fresh Chrome session per page load to avoid Cloudflare blocking.
Each session makes exactly ONE request, guaranteeing CF never tracks between requests.

Usage:
  python scraper.py --years 2015-2025 --problems 1-25 --contest A
  python scraper.py --years 2010-2020 --contest both --output my_problems.jsonl
  python scraper.py --years 2019,2021,2023 --problems 11-25 --contest B
"""

import argparse
import json
import sys
import time
import re
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup, Tag, NavigableString

YEAR_URL = "https://artofproblemsolving.com/wiki/index.php/{year}_AMC_10{contest}_Problems"
ANSWER_KEY_URL = "https://artofproblemsolving.com/wiki/index.php/{year}_AMC_10{contest}_Answer_Key"
PROBLEM_URL = "https://artofproblemsolving.com/wiki/index.php/{year}_AMC_10{contest}_Problems/Problem_{num}"

ANSWER_LETTERS = set("ABCDE")


def _parse_range(s, label):
    """Parse '2015-2025' → [2015..2025] or '3,5,7' → [3, 5, 7]."""
    s = s.strip()
    if "," not in s and "-" in s:
        lo, _, hi = s.partition("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def p(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def _classify(problem_num):
    if problem_num <= 10:
        return "1-10"
    elif problem_num <= 20:
        return "11-20"
    else:
        return "21-25"


def _make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )
    opts.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })
    driver.set_page_load_timeout(45)
    return driver


def _fetch_page(url, wait=8):
    """Spin up a fresh Chrome session, load one page, return HTML, then quit."""
    driver = _make_driver()
    html = None
    try:
        driver.get(url)
        time.sleep(wait + random.uniform(0.5, 2))
        title = driver.title
        if "Cloudflare" in title or "Attention Required" in title:
            p(f"    CF block, waiting 20s and retrying...")
            time.sleep(20)
            src = driver.page_source
            if "mw-parser-output" in src:
                html = src
            else:
                time.sleep(15)
                src2 = driver.page_source
                if "mw-parser-output" in src2:
                    html = src2
        elif "mw-parser-output" in driver.page_source:
            html = driver.page_source
        else:
            p(f"    No content (title: {title[:50]})")
    except Exception as e:
        p(f"    Error: {e}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    return html


def _text_with_latex(element):
    """Walk a BeautifulSoup element, substituting LaTeX <img> alt text inline."""
    if isinstance(element, NavigableString):
        return str(element)
    result = []
    for node in element.descendants:
        if isinstance(node, Tag):
            if node.name == "img" and "latex" in node.get("class", []):
                result.append(node.get("alt", ""))
        else:
            if node.parent and node.parent.name != "img":
                result.append(str(node))
    return "".join(result)


def _parse_year_page(soup):
    """Extract all problems from a full-year AoPS wiki page."""
    content = soup.find("div", class_="mw-parser-output")
    if not content:
        return {}
    problems = {}
    for h in content.find_all(["h2", "h3"]):
        m = re.match(r"^Problem\s+(\d+)$", h.get_text(strip=True), re.IGNORECASE)
        if not m:
            continue
        num = int(m.group(1))
        if num not in range(1, 26):
            continue
        parts = []
        for sib in h.next_siblings:
            if isinstance(sib, Tag) and sib.name in ["h2", "h3"]:
                sib_txt = sib.get_text(strip=True)
                if re.match(r"^Problem\s*\d+", sib_txt) or "See Also" in sib_txt:
                    break
            if isinstance(sib, Tag):
                t = _text_with_latex(sib).strip()
                if t:
                    parts.append(t)
        if parts:
            problems[num] = "\n".join(parts).strip()
    return problems


def _fetch_answer_key(year, contest):
    url = ANSWER_KEY_URL.format(year=year, contest=contest)
    html = _fetch_page(url, wait=7)
    if not html:
        return {}
    content = BeautifulSoup(html, "lxml").find("div", class_="mw-parser-output")
    if not content:
        return {}
    lines = [l.strip() for l in content.get_text(separator="\n", strip=True).split("\n") if l.strip()]
    answers = {}
    prob_num = 1
    for line in lines:
        if line in ANSWER_LETTERS:
            answers[prob_num] = line
            prob_num += 1
            if prob_num > 25:
                break
    if len(answers) < 20:
        answers = {}
        for line in lines:
            m = re.match(r"(\d+)[.):]\s*([A-E])\b", line)
            if m:
                answers[int(m.group(1))] = m.group(2)
    return answers


def _fetch_solution(year, contest, num):
    """Fetch solution text from a problem's individual AoPS page."""
    url = PROBLEM_URL.format(year=year, contest=contest, num=num)
    html = _fetch_page(url, wait=8)
    if not html:
        return None
    content = BeautifulSoup(html, "lxml").find("div", class_="mw-parser-output")
    if not content:
        return None
    sol_heading = None
    for h in content.find_all(["h2", "h3"]):
        if re.match(r"^Solution", h.get_text(strip=True), re.IGNORECASE):
            sol_heading = h
            break
    if not sol_heading:
        return None
    parts = []
    for sib in sol_heading.next_siblings:
        if isinstance(sib, Tag) and sib.name in ["h2", "h3"]:
            if any(kw in sib.get_text(strip=True) for kw in ["See Also", "See also", "Video Solution", "Video solution"]):
                break
        if isinstance(sib, Tag):
            t = _text_with_latex(sib).strip()
            if t:
                parts.append(t)
    return "\n".join(parts).strip() if parts else None


def _fetch_individual_problem(year, contest, num):
    url = PROBLEM_URL.format(year=year, contest=contest, num=num)
    html = _fetch_page(url, wait=8)
    if not html:
        return None
    content = BeautifulSoup(html, "lxml").find("div", class_="mw-parser-output")
    if not content:
        return None
    ph = None
    for h in content.find_all(["h2", "h3"]):
        if re.match(r"^Problem\s*\d*$", h.get_text(strip=True), re.IGNORECASE):
            ph = h
            break
    if not ph:
        return None
    parts = []
    for sib in ph.next_siblings:
        if isinstance(sib, Tag) and sib.name in ["h2", "h3"]:
            if any(kw in sib.get_text(strip=True) for kw in ["Solution", "Video", "See Also", "Hint"]):
                break
        if isinstance(sib, Tag):
            t = _text_with_latex(sib).strip()
            if t:
                parts.append(t)
    return "\n".join(parts).strip() if parts else None


def _str_to_bool(v):
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (True or False)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape AMC 10 problems from AoPS wiki.")
    parser.add_argument("--years", default="2015-2025",
                        help="Year range or comma-list, e.g. '2015-2025' or '2015,2016,2020' (default: 2015-2025)")
    parser.add_argument("--problems", default="1-25",
                        help="Problem range or comma-list, e.g. '1-25' or '3,7,15' (default: 1-25)")
    parser.add_argument("--contest", choices=["A", "B", "both"], default="A",
                        help="Contest to scrape: A, B, or both (default: A)")
    parser.add_argument("--output", default="all_problems.jsonl",
                        help="Output JSONL file (default: all_problems.jsonl)")
    parser.add_argument("--solutions-only", action="store_true",
                        help="Fetch solutions and add them to an existing JSONL file (reads --output file)")
    parser.add_argument("--solutions", type=_str_to_bool, default=False, metavar="True|False",
                        help="Include solutions when creating the dataset (default: False)")
    args = parser.parse_args()

    # --solutions-only: read existing JSONL, fetch solutions, write back
    if args.solutions_only:
        input_file = args.output
        problems = []
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        p(f"Loaded {len(problems)} problems from {input_file}")
        p("Fetching solutions (one fresh Chrome session per problem)...\n")
        for i, prob in enumerate(problems):
            if prob.get("solution"):
                p(f"  [{i+1}/{len(problems)}] {prob['year']} 10{prob['contest']} P{prob['problem_num']}: already has solution, skipping")
                continue
            year, contest, num = prob["year"], prob["contest"], prob["problem_num"]
            p(f"  [{i+1}/{len(problems)}] {year} 10{contest} P{num}...", end=" ")
            sol = _fetch_solution(year, contest, num)
            prob["solution"] = sol
            p("OK" if sol else "FAILED")
            # Checkpoint every 10 problems so progress survives interruption
            if (i + 1) % 10 == 0:
                with open(input_file, "w") as f:
                    for pr in problems:
                        f.write(json.dumps(pr) + "\n")
            time.sleep(random.uniform(2, 4))
        with open(input_file, "w") as f:
            for prob in problems:
                f.write(json.dumps(prob) + "\n")
        p(f"\nUpdated {input_file}")
        p(f"With solutions: {sum(1 for pr in problems if pr.get('solution'))}/{len(problems)}")
        sys.exit(0)

    years = _parse_range(args.years, "years")
    problem_nums = _parse_range(args.problems, "problems")
    contests = ["A", "B"] if args.contest == "both" else [args.contest]

    p(f"Scraping AMC 10{args.contest} | Years: {years[0]}–{years[-1]} | Problems: {problem_nums[0]}–{problem_nums[-1]}")
    p("Strategy: one fresh Chrome session per page (avoids Cloudflare)\n")

    collected = {}  # (year, contest, num) -> dict

    for contest in contests:
        for year in years:
            p(f"\n=== {year} AMC 10{contest} ===")

            answer_key = _fetch_answer_key(year, contest)
            p(f"  Got {len(answer_key)}/25 answers")
            time.sleep(random.uniform(2, 4))

            html = _fetch_page(YEAR_URL.format(year=year, contest=contest), wait=8)
            year_probs = _parse_year_page(BeautifulSoup(html, "lxml")) if html else {}
            p(f"  Got {len(year_probs)}/25 problems from year page")
            time.sleep(random.uniform(2, 4))

            for num in [n for n in problem_nums if n not in year_probs]:
                p(f"    Individual: P{num}...", end=" ")
                text = _fetch_individual_problem(year, contest, num)
                if text:
                    year_probs[num] = text
                    p("OK")
                else:
                    p("FAILED")
                time.sleep(random.uniform(2, 4))

            for num in problem_nums:
                text = year_probs.get(num, "")
                if len(text) >= 15:
                    collected[(year, contest, num)] = {
                        "year": year,
                        "contest": contest,
                        "problem_num": num,
                        "class": _classify(num),
                        "problem": text,
                        "answer": answer_key.get(num),
                        "url": PROBLEM_URL.format(year=year, contest=contest, num=num),
                    }

            got = sum(1 for n in problem_nums if (year, contest, n) in collected)
            p(f"  Stored {got}/{len(problem_nums)}")
            time.sleep(random.uniform(3, 6))

    # Second pass: retry anything still missing
    missing = [
        (y, c, n)
        for c in contests
        for y in years
        for n in problem_nums
        if (y, c, n) not in collected
    ]
    if missing:
        p(f"\n=== Second pass: {len(missing)} missing ===")
        for year, contest, num in missing:
            p(f"  Retry {year} 10{contest} P{num}...", end=" ")
            text = _fetch_individual_problem(year, contest, num)
            if text:
                answer_key = _fetch_answer_key(year, contest)
                collected[(year, contest, num)] = {
                    "year": year,
                    "contest": contest,
                    "problem_num": num,
                    "class": _classify(num),
                    "problem": text,
                    "answer": answer_key.get(num),
                    "url": PROBLEM_URL.format(year=year, contest=contest, num=num),
                }
                p("OK")
            else:
                p("STILL MISSING")
            time.sleep(random.uniform(2, 5))

    if args.solutions:
        p(f"\n=== Fetching solutions for {len(collected)} problems ===")
        for i, (key, prob) in enumerate(collected.items()):
            year, contest, num = key
            p(f"  [{i+1}/{len(collected)}] {year} 10{contest} P{num}...", end=" ")
            sol = _fetch_solution(year, contest, num)
            prob["solution"] = sol
            p("OK" if sol else "FAILED")
            time.sleep(random.uniform(2, 4))

    sorted_problems = [collected[k] for k in sorted(collected.keys())]
    total_expected = len(years) * len(contests) * len(problem_nums)

    p(f"\n=== Final Results ===")
    p(f"Fetched: {len(sorted_problems)}/{total_expected}")

    final_missing = [(y, c, n) for c in contests for y in years for n in problem_nums if (y, c, n) not in collected]
    if final_missing:
        p(f"Missing: {final_missing}")

    with open(args.output, "w") as f:
        for prob in sorted_problems:
            f.write(json.dumps(prob) + "\n")

    p(f"Wrote {len(sorted_problems)} problems to {args.output}")

    by_class = {}
    for prob in sorted_problems:
        c = prob["class"]
        by_class[c] = by_class.get(c, 0) + 1
    p(f"Distribution: {by_class}")
    p(f"With answers: {sum(1 for p in sorted_problems if p.get('answer'))}/{len(sorted_problems)}")
