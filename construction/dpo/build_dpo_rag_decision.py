# -*- coding: utf-8 -*-
import json
import re
import random
from pathlib import Path
from collections import Counter, defaultdict


# =========================
# Config
# =========================

SOURCE_DIR = Path("data/interim/dpo_sources")
QA_PATH = SOURCE_DIR / "qa_train_dedup_selected.jsonl"
MCQ_PATH = SOURCE_DIR / "mcq_dedup_selected_v2.jsonl"

OUT_DIR = Path("data/dpo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "dpo_rag_decision.jsonl"
REPORT_PATH = OUT_DIR / "dpo_rag_decision_report.json"

SEED = 42
random.seed(SEED)

TARGETS = {
    "rag_continue_decision": 140,
    "rag_stop_decision": 60,
    "rag_final_answer_repair": 40,
    "mcq_guard": 60,
}

CONTINUE_RECALL_MAX = 0.60
STOP_RECALL_MIN = 0.75

MAX_LEN_RATIO_FINAL_REPAIR = 1.30


# =========================
# Basic utils
# =========================

# 关键修复：使用非捕获组，避免 findall 只返回 fact/case/diag
FACT_RE = re.compile(
    r"(?:fact|case|diag)_[a-zA-Z0-9]+_[0-9]{3}_[0-9]{3}|diag_manual_[0-9]{3}"
)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] bad json line {line_no}: {e}")
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_fact_ids(text: str):
    if not text:
        return []
    return sorted(set(m.group(0) for m in FACT_RE.finditer(text)))


def is_tag_answer(text):
    if not text:
        return False
    t = text.strip()
    return (
        t.startswith("<plan>")
        or t.startswith("<query>")
        or t.startswith("<judge>")
        or t.endswith("</plan>")
        or t.endswith("</query>")
        or t.endswith("</judge>")
    )


def extract_user_question(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""


def get_final_answer(messages):
    for m in reversed(messages):
        if m.get("role") == "assistant":
            c = m.get("content", "").strip()
            if c and not is_tag_answer(c):
                return c
    return ""


def parse_tool_content(content):
    try:
        arr = json.loads(content)
        if not isinstance(arr, list):
            return []
        out = []
        for x in arr:
            if not isinstance(x, dict):
                continue
            card_id = x.get("card_id", "")
            if not card_id:
                continue
            out.append({
                "card_id": card_id,
                "refined_result": x.get("refined_result", ""),
                "citation_text": x.get("citation_text", ""),
                "query": x.get("query", ""),
            })
        return out
    except Exception:
        return []


def extract_tag(text, tag):
    if not text:
        return ""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return ""


def find_next_query(messages, start_idx):
    for j in range(start_idx + 1, len(messages)):
        m = messages[j]
        if m.get("role") != "assistant":
            continue

        q = extract_tag(m.get("content", ""), "query")
        if q:
            return q

        c = m.get("content", "").strip()
        if c and not is_tag_answer(c):
            return ""
    return ""


def find_next_judge(messages, start_idx):
    for j in range(start_idx + 1, len(messages)):
        m = messages[j]
        if m.get("role") != "assistant":
            continue

        judge = extract_tag(m.get("content", ""), "judge")
        if judge:
            return judge

        c = m.get("content", "").strip()
        if c and not is_tag_answer(c):
            return ""
    return ""


def collect_tool_steps(row):
    messages = row.get("messages", [])
    steps = []
    cumulative = []

    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue

        cards = parse_tool_content(m.get("content", ""))
        if not cards:
            continue

        cumulative.extend(cards)
        cum_ids = sorted(set(x["card_id"] for x in cumulative))

        next_query = find_next_query(messages, i)
        next_judge = find_next_judge(messages, i)

        steps.append({
            "tool_index": i,
            "current_cards": cards,
            "cumulative_cards": list(cumulative),
            "cumulative_fact_ids": cum_ids,
            "next_query": next_query,
            "next_judge": next_judge,
        })

    return steps


def gold_ids_for_row(row):
    ids = row.get("citations") or []
    if not ids:
        ids = row.get("seed_card_ids") or []
    return sorted(set(ids))


def recall(cited, gold):
    gold = set(gold)
    if not gold:
        return 0.0
    return len(set(cited) & gold) / len(gold)


def precision(cited, gold):
    cited = set(cited)
    if not cited:
        return 0.0
    return len(cited & set(gold)) / len(cited)


def format_cards(cards, max_cards=12):
    lines = []
    seen = set()

    for c in cards:
        cid = c.get("card_id", "")
        if not cid or cid in seen:
            continue

        seen.add(cid)
        text = c.get("refined_result") or c.get("citation_text") or ""
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 180:
            text = text[:180] + "..."

        lines.append(f"{cid}: {text}")

        if len(lines) >= max_cards:
            break

    return "\n".join(lines)


def make_decision_prompt(row, step):
    messages = row.get("messages", [])
    user_q = extract_user_question(messages)
    cards_text = format_cards(step["cumulative_cards"], max_cards=12)

    return (
        "你是一个面向藏医知识问答的检索决策助手。\n"
        "请根据当前问题与已检索到的知识卡片，判断证据是否充分。\n"
        "如果证据不足，必须输出 <judge>证据不足，继续检索...</judge> 并给出 <query>...</query>。\n"
        "如果证据充分，只输出 <judge>证据充分，可以回答。</judge>。\n\n"
        f"【用户问题】\n{user_q}\n\n"
        f"【当前已检索知识卡片】\n{cards_text}\n\n"
        "请输出下一步检索决策。"
    )


def make_final_answer_prompt(row):
    messages = row.get("messages", [])
    user_q = extract_user_question(messages)
    steps = collect_tool_steps(row)

    all_cards = []
    for s in steps:
        all_cards.extend(s["current_cards"])

    cards_text = format_cards(all_cards, max_cards=16)

    return (
        "请根据以下知识卡片回答问题，并在答案中引用支持依据。\n\n"
        f"【知识卡片】\n{cards_text}\n\n"
        f"【问题】\n{user_q}\n"
    )


def clean_template_tail(text):
    if not text:
        return ""

    out = text.strip()

    patterns = [
        r"以上建议仅供参考.*$",
        r"需要提醒的是.*?专业医师.*$",
        r"请在专业医师指导下.*$",
        r"建议及时就医.*$",
        r"具体治疗方案请在专业医师指导下.*$",
    ]

    for p in patterns:
        out = re.sub(p, "", out, flags=re.S).strip()

    return out


def remove_some_gold_citations(answer, gold_ids, keep_min=1):
    cited = extract_fact_ids(answer)
    gold_set = set(gold_ids)
    gold_cited = [x for x in cited if x in gold_set]

    if len(gold_cited) <= keep_min:
        return ""

    remove_n = max(1, min(2, len(gold_cited) - keep_min))
    to_remove = set(gold_cited[-remove_n:])

    rejected = answer

    for fid in to_remove:
        rejected = rejected.replace(fid, "")

    rejected = re.sub(r"\(\s*,\s*", "(", rejected)
    rejected = re.sub(r",\s*\)", ")", rejected)
    rejected = re.sub(r"（\s*,\s*", "（", rejected)
    rejected = re.sub(r",\s*）", "）", rejected)
    rejected = re.sub(r"\(\s*\)", "", rejected)
    rejected = re.sub(r"（\s*）", "", rejected)
    rejected = re.sub(r"《\s*》", "", rejected)
    rejected = re.sub(r"\[\s*\]", "", rejected)
    rejected = re.sub(r"\s+", " ", rejected).strip()

    if rejected == answer:
        return ""

    return rejected


def pair_id(prefix, idx):
    return f"{prefix}_{idx:04d}"


# =========================
# Build RAG decision pairs
# =========================

def build_continue_pairs(qa_rows, target):
    candidates = []

    for row_idx, row in enumerate(qa_rows):
        gold = gold_ids_for_row(row)
        if len(gold) < 2:
            continue

        steps = collect_tool_steps(row)
        if len(steps) < 2:
            continue

        for step_idx, step in enumerate(steps[:-1]):
            cur_ids = step["cumulative_fact_ids"]
            r = recall(cur_ids, gold)

            if r >= CONTINUE_RECALL_MAX:
                continue

            next_query = step.get("next_query", "").strip()
            if not next_query:
                continue

            later_ids = set()
            for later in steps[step_idx + 1:]:
                later_ids.update(later["cumulative_fact_ids"])

            # 后续检索必须能补到至少一个 gold fact
            if len((later_ids - set(cur_ids)) & set(gold)) == 0:
                continue

            prompt = make_decision_prompt(row, step)

            chosen = (
                "<judge>证据不足，继续检索。当前证据只覆盖了问题的部分要点，"
                "仍缺少关键典籍依据，不能直接作答。</judge>\n"
                f"<query>{next_query}</query>"
            )

            rejected = "<judge>证据充分，可以回答。</judge>"

            margin = round((1.0 - r) * 0.25, 4)

            candidates.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "type": "rag_continue_decision",
                "meta": {
                    "source": "qa_trajectory",
                    "source_idx": row_idx,
                    "question_type": row.get("question_type", ""),
                    "format_id": row.get("format_id", ""),
                    "error_type": "premature_stop_when_evidence_insufficient",
                    "step_idx": step_idx,
                    "rounds": row.get("rounds"),
                    "gold_fact_ids": gold,
                    "current_fact_ids": cur_ids,
                    "current_gold_recall": round(r, 4),
                    "next_query": next_query,
                    "margin": margin,
                }
            })

    candidates.sort(key=lambda x: (
        abs(x["meta"]["current_gold_recall"] - 0.35),
        len(x["prompt"])
    ))

    selected = []
    used = set()

    for c in candidates:
        key = c["prompt"][:400]
        if key in used:
            continue
        used.add(key)
        selected.append(c)
        if len(selected) >= target:
            break

    if len(selected) < target:
        print(f"[WARN] continue need {target}, got {len(selected)}")

    return selected


def build_stop_pairs(qa_rows, target):
    candidates = []

    for row_idx, row in enumerate(qa_rows):
        gold = gold_ids_for_row(row)
        if len(gold) < 1:
            continue

        steps = collect_tool_steps(row)
        if not steps:
            continue

        for step_idx, step in enumerate(steps):
            cur_ids = step["cumulative_fact_ids"]
            r = recall(cur_ids, gold)

            if r < STOP_RECALL_MIN:
                continue

            prompt = make_decision_prompt(row, step)

            chosen = "<judge>证据充分，可以回答。</judge>"

            rejected = (
                "<judge>证据不足，继续检索。当前证据仍不足，需要继续查找相关典籍依据。</judge>\n"
                "<query>继续检索与问题核心要点相关的典籍依据</query>"
            )

            margin = round((r - STOP_RECALL_MIN) * 0.25 + 0.08, 4)

            candidates.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "type": "rag_stop_decision",
                "meta": {
                    "source": "qa_trajectory",
                    "source_idx": row_idx,
                    "question_type": row.get("question_type", ""),
                    "format_id": row.get("format_id", ""),
                    "error_type": "unnecessary_continue_when_evidence_sufficient",
                    "step_idx": step_idx,
                    "rounds": row.get("rounds"),
                    "gold_fact_ids": gold,
                    "current_fact_ids": cur_ids,
                    "current_gold_recall": round(r, 4),
                    "margin": margin,
                }
            })

    candidates.sort(key=lambda x: (
        abs(x["meta"]["current_gold_recall"] - 0.80),
        len(x["prompt"])
    ))

    selected = []
    used = set()

    for c in candidates:
        key = c["prompt"][:400]
        if key in used:
            continue
        used.add(key)
        selected.append(c)
        if len(selected) >= target:
            break

    if len(selected) < target:
        print(f"[WARN] stop need {target}, got {len(selected)}")

    return selected


# =========================
# Build final answer repair
# =========================

def build_final_answer_repair(qa_rows, target):
    candidates = []

    for row_idx, row in enumerate(qa_rows):
        gold = gold_ids_for_row(row)
        if len(gold) < 2:
            continue

        ans = clean_template_tail(get_final_answer(row.get("messages", [])))
        if not ans:
            continue

        cited = extract_fact_ids(ans)
        if not cited:
            continue

        r = recall(cited, gold)
        p = precision(cited, gold)

        if r < 0.75 or p < 0.90:
            continue

        rejected = remove_some_gold_citations(
            ans,
            gold,
            keep_min=max(1, len(gold) // 2)
        )

        if not rejected:
            continue

        cr = recall(extract_fact_ids(ans), gold)
        rr = recall(extract_fact_ids(rejected), gold)

        if cr <= rr:
            continue

        if len(ans) > MAX_LEN_RATIO_FINAL_REPAIR * max(1, len(rejected)):
            continue

        prompt = make_final_answer_prompt(row)

        candidates.append({
            "prompt": prompt,
            "chosen": ans,
            "rejected": rejected,
            "type": "rag_final_answer_repair",
            "meta": {
                "source": "qa_trajectory",
                "source_idx": row_idx,
                "question_type": row.get("question_type", ""),
                "format_id": row.get("format_id", ""),
                "error_type": "missing_gold_citation_in_final_answer",
                "gold_fact_ids": gold,
                "chosen_fact_ids": extract_fact_ids(ans),
                "rejected_fact_ids": extract_fact_ids(rejected),
                "chosen_gold_recall": round(cr, 4),
                "rejected_gold_recall": round(rr, 4),
                "chosen_citation_precision": round(precision(extract_fact_ids(ans), gold), 4),
                "rejected_citation_precision": round(precision(extract_fact_ids(rejected), gold), 4),
                "len_ratio_chosen_over_rejected": round(len(ans) / max(1, len(rejected)), 4),
                "margin": round((cr - rr) * 0.3, 4),
            }
        })

    candidates.sort(key=lambda x: (
        -x["meta"]["chosen_gold_recall"],
        abs(x["meta"]["len_ratio_chosen_over_rejected"] - 1.05)
    ))

    selected = candidates[:target]

    if len(selected) < target:
        print(f"[WARN] final repair need {target}, got {len(selected)}")

    return selected


# =========================
# MCQ guard
# =========================

def extract_mcq_prompt(row):
    messages = row.get("messages", [])
    prompt = ""

    for m in messages:
        if m.get("role") == "user":
            prompt = m.get("content", "").strip()
            break

    if not prompt:
        return ""

    # 关键修复：删除 E 选项行，适配四选项评测
    lines = prompt.splitlines()
    new_lines = []

    for line in lines:
        if re.match(r"^\s*E[\.．、]\s*", line):
            continue
        new_lines.append(line)

    return "\n".join(new_lines).strip()


def choose_wrong_letter(gold):
    letters = ["A", "B", "C", "D"]
    choices = [x for x in letters if x != gold]
    return random.choice(choices) if choices else "A"


def build_mcq_guard(mcq_rows, target):
    """
    构造均衡 MCQ guard。
    目标：
    - target=60 时，A/B/C/D 各 15 条
    - 每个选项内部再分配三类 error_type：
        correct_vs_wrong_option
        option_only_vs_explanation
        single_option_vs_multiple_uncertain
    """

    letters = ["A", "B", "C", "D"]

    # target=60 -> 每个 gold 15 条
    per_gold = target // 4
    remainder = target % 4

    gold_targets = {}
    for i, l in enumerate(letters):
        gold_targets[l] = per_gold + (1 if i < remainder else 0)

    # 每个 gold 内部三种类型大致均衡
    # per_gold=15 -> 5/5/5
    def split_type_counts(n):
        base = n // 3
        rem = n % 3
        types = [
            "correct_vs_wrong_option",
            "option_only_vs_explanation",
            "single_option_vs_multiple_uncertain",
        ]
        out = {}
        for i, t in enumerate(types):
            out[t] = base + (1 if i < rem else 0)
        return out

    # 按 gold 分桶
    by_gold = defaultdict(list)

    for idx, row in enumerate(mcq_rows):
        gold = str(row.get("gold", "")).strip().upper()
        if gold not in letters:
            continue

        prompt = extract_mcq_prompt(row)
        if not prompt:
            continue

        # 确保 E 已被删掉
        if re.search(r"^\s*E[\.．、]\s*", prompt, flags=re.M):
            continue

        by_gold[gold].append((idx, row, prompt, gold))

    for g in letters:
        random.shuffle(by_gold[g])

    rows = []

    for gold in letters:
        need = gold_targets[gold]
        type_targets = split_type_counts(need)

        candidates = by_gold[gold]
        if len(candidates) < need:
            print(f"[WARN] MCQ gold={gold} need {need}, got candidate {len(candidates)}")

        ptr = 0

        for error_type, n in type_targets.items():
            for _ in range(n):
                if ptr >= len(candidates):
                    break

                idx, row, prompt, gold = candidates[ptr]
                ptr += 1

                wrong = choose_wrong_letter(gold)

                if error_type == "correct_vs_wrong_option":
                    pair = {
                        "prompt": prompt,
                        "chosen": gold,
                        "rejected": wrong,
                        "type": "mcq_guard",
                        "meta": {
                            "source": "mcq_trajectory",
                            "source_idx": idx,
                            "error_type": "correct_vs_wrong_option",
                            "gold": gold,
                            "wrong": wrong,
                            "margin": 0.0,
                        }
                    }

                elif error_type == "option_only_vs_explanation":
                    pair = {
                        "prompt": prompt,
                        "chosen": gold,
                        "rejected": f"答案是{gold}。因为根据题干和选项分析，{gold}更符合相关藏医典籍知识。",
                        "type": "mcq_guard",
                        "meta": {
                            "source": "mcq_trajectory",
                            "source_idx": idx,
                            "error_type": "option_only_vs_explanation",
                            "gold": gold,
                            "margin": 0.0,
                        }
                    }

                elif error_type == "single_option_vs_multiple_uncertain":
                    pair = {
                        "prompt": prompt,
                        "chosen": gold,
                        "rejected": f"{gold}或{wrong}",
                        "type": "mcq_guard",
                        "meta": {
                            "source": "mcq_trajectory",
                            "source_idx": idx,
                            "error_type": "single_option_vs_multiple_uncertain",
                            "gold": gold,
                            "wrong": wrong,
                            "margin": 0.0,
                        }
                    }

                else:
                    continue

                rows.append(pair)

    # 如果因为某些 gold 不够导致不足，用剩余样本补齐，但仍优先补少的 gold
    if len(rows) < target:
        current_dist = Counter(r["meta"]["gold"] for r in rows)
        used_indices = set(r["meta"]["source_idx"] for r in rows)

        leftovers = []
        for gold in letters:
            for idx, row, prompt, g in by_gold[gold]:
                if idx not in used_indices:
                    leftovers.append((idx, row, prompt, g))

        random.shuffle(leftovers)

        while len(rows) < target and leftovers:
            # 优先选择当前最少的 gold
            current_dist = Counter(r["meta"]["gold"] for r in rows)
            min_gold = min(letters, key=lambda x: current_dist.get(x, 0))

            chosen_item = None
            for i, item in enumerate(leftovers):
                if item[3] == min_gold:
                    chosen_item = leftovers.pop(i)
                    break

            if chosen_item is None:
                chosen_item = leftovers.pop()

            idx, row, prompt, gold = chosen_item
            wrong = choose_wrong_letter(gold)

            rows.append({
                "prompt": prompt,
                "chosen": gold,
                "rejected": wrong,
                "type": "mcq_guard",
                "meta": {
                    "source": "mcq_trajectory",
                    "source_idx": idx,
                    "error_type": "correct_vs_wrong_option",
                    "gold": gold,
                    "wrong": wrong,
                    "margin": 0.0,
                    "fallback_fill": True,
                }
            })

    if len(rows) < target:
        print(f"[WARN] mcq need {target}, got {len(rows)}")

    rows = rows[:target]

    final_dist = Counter(r["meta"]["gold"] for r in rows)
    print(f"[INFO] MCQ balanced gold distribution: {dict(final_dist)}")

    return rows

# =========================
# Report
# =========================

def summarize(rows):
    type_counts = Counter(r.get("type", "unknown") for r in rows)
    err_counts = Counter(r.get("meta", {}).get("error_type", "unknown") for r in rows)

    margins = []
    continue_recalls = []
    stop_recalls = []
    final_chosen_recalls = []
    final_rejected_recalls = []
    final_len_ratios = []
    mcq_gold = Counter()

    for r in rows:
        m = r.get("meta", {})

        if "margin" in m:
            margins.append(m["margin"])

        if r.get("type") == "rag_continue_decision" and "current_gold_recall" in m:
            continue_recalls.append(m["current_gold_recall"])

        if r.get("type") == "rag_stop_decision" and "current_gold_recall" in m:
            stop_recalls.append(m["current_gold_recall"])

        if r.get("type") == "rag_final_answer_repair":
            if "chosen_gold_recall" in m:
                final_chosen_recalls.append(m["chosen_gold_recall"])
            if "rejected_gold_recall" in m:
                final_rejected_recalls.append(m["rejected_gold_recall"])
            if "len_ratio_chosen_over_rejected" in m:
                final_len_ratios.append(m["len_ratio_chosen_over_rejected"])

        if r.get("type") == "mcq_guard":
            g = m.get("gold")
            if g:
                mcq_gold[g] += 1

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "total": len(rows),
        "type_counts": dict(type_counts),
        "error_type_counts": dict(err_counts),
        "avg_margin": avg(margins),
        "continue_avg_current_gold_recall": avg(continue_recalls),
        "stop_avg_current_gold_recall": avg(stop_recalls),
        "final_repair_avg_chosen_gold_recall": avg(final_chosen_recalls),
        "final_repair_avg_rejected_gold_recall": avg(final_rejected_recalls),
        "final_repair_avg_len_ratio": avg(final_len_ratios),
        "mcq_gold_distribution": dict(mcq_gold),
    }


def dedup_pairs(rows):
    seen = set()
    out = []
    removed = 0

    for r in rows:
        key = (r["prompt"], r["chosen"], r["rejected"])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(r)

    return out, removed


# =========================
# Main
# =========================

def main():
    qa_rows = read_jsonl(QA_PATH)
    mcq_rows = read_jsonl(MCQ_PATH)

    print(f"[INFO] Loaded QA rows: {len(qa_rows)}")
    print(f"[INFO] Loaded MCQ rows: {len(mcq_rows)}")

    continue_pairs = build_continue_pairs(
        qa_rows,
        TARGETS["rag_continue_decision"]
    )
    print(f"[INFO] RAG continue decision: {len(continue_pairs)}")

    stop_pairs = build_stop_pairs(
        qa_rows,
        TARGETS["rag_stop_decision"]
    )
    print(f"[INFO] RAG stop decision: {len(stop_pairs)}")

    final_pairs = build_final_answer_repair(
        qa_rows,
        TARGETS["rag_final_answer_repair"]
    )
    print(f"[INFO] RAG final answer repair: {len(final_pairs)}")

    mcq_pairs = build_mcq_guard(
        mcq_rows,
        TARGETS["mcq_guard"]
    )
    print(f"[INFO] MCQ guard: {len(mcq_pairs)}")

    all_rows = []
    all_rows.extend(continue_pairs)
    all_rows.extend(stop_pairs)
    all_rows.extend(final_pairs)
    all_rows.extend(mcq_pairs)

    random.shuffle(all_rows)

    all_rows, removed = dedup_pairs(all_rows)

    counters = Counter()
    for r in all_rows:
        t = r.get("type", "pair")
        counters[t] += 1
        r["id"] = pair_id(t, counters[t])

    report = summarize(all_rows)
    report["dedup_removed_pairs"] = removed
    report["targets"] = TARGETS

    write_jsonl(OUT_PATH, all_rows)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n[INFO] Done.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nsaved: {OUT_PATH}")
    print(f"saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
