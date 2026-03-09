import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag import answer_query

GOLDEN_PATH = ROOT / "eval" / "golden.jsonl"

def load_golden():
    tests = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tests.append(json.loads(line))
    return tests

def contains_all(text: str, items):
    t = text.lower()
    return all(x.lower() in t for x in items)

def cites_any(citations, required_chunk_ids):
    found = {c["chunk_id"] for c in citations}
    return any(x in found for x in required_chunk_ids)

def main():
    tests = load_golden()
    passed = 0
    failed = 0

    for t in tests:
        qid = t.get("id", "?")
        q = t["question"]
        must_include = t.get("must_include", [])
        must_cite = t.get("must_cite", [])

        out = answer_query(q, top_k=6)
        ans = out.get("answer", "")
        citations = out.get("citations", [])

        ok_include = contains_all(ans, must_include) if must_include else True
        ok_cite = cites_any(citations, must_cite) if must_cite else True

        if ok_include and ok_cite:
            passed += 1
            print(f"[PASS] {qid}: {q}")
        else:
            failed += 1
            print(f"[FAIL] {qid}: {q}")
            print("  Answer:", ans)
            print("  Citations:", [c["chunk_id"] for c in citations])
            print("  must_include:", must_include, "ok:", ok_include)
            print("  must_cite:", must_cite, "ok:", ok_cite)

    total = passed + failed
    score = passed / total if total else 0.0
    print(f"\nScore: {passed}/{total} = {score:.2f}")

    if score < 0.80:
        print("❌ Quality gate failed (score < 0.80)")
        sys.exit(1)

    print("✅ Quality gate passed")
    sys.exit(0)

if __name__ == "__main__":
    main()
