"""
gate_check.py  (Person 3)
Checks whether the MAX_DISTANCE cut-off separates in-scope from out-of-scope questions.
No LLM is called, so it uses no API quota and runs in seconds.

    py gate_check.py                     # uses MAX_DISTANCE from the environment, else 0.85
    $env:MAX_DISTANCE="0.80"; py gate_check.py

The "tuning" set is the 16 questions the threshold was chosen from, so its result proves
nothing. The "validation" set was written afterwards: only that one counts as evidence.
"""
import os

import llm_answering
from llm_answering import TEST_QUESTIONS, get_chunks

# Read BEFORE the gate is switched off below. If MAX_DISTANCE=0 (gate disabled) this script
# still checks the default 0.85 so you can see what the gate WOULD do.
THRESHOLD = llm_answering.MAX_DISTANCE or 0.85
if not llm_answering.MAX_DISTANCE:
    print("Note: MAX_DISTANCE is 0 (gate disabled in the app); checking the default 0.85 here.")
# Measure the REAL distances: switch the gate off inside llm_answering, otherwise it would
# return no chunks for out-of-scope questions and every distance would show as inf.
llm_answering.MAX_DISTANCE = 0.0
TOP_K = int(os.environ.get("TOP_K", "7"))

# Written after the threshold was chosen. Every in-scope topic was checked to exist in the
# notes, and every out-of-scope topic was checked to be absent from them.
VALIDATION = [
    ("What is a DDoS attack?", True),
    ("What is defence in depth?", True),
    ("What is subnetting?", True),
    ("What is a digital signature used for?", True),
    ("What is social engineering?", True),
    ("What is the purpose of DHCP?", True),
    ("What is RAID 5 and how does it work?", False),
    ("How do I install Docker on Windows?", False),
    ("What is SOC 2 compliance?", False),
    ("Give me a recipe for chocolate cake.", False),
    ("Who is the best football player of all time?", False),
    ("Explain how photosynthesis works.", False),
]

# Student-style phrasing (vague, indirect). Topics are all in the notes.
STRESS = [
    ("Why shouldn't everyone just have admin access?", True),             # least privilege
    ("How do hackers get into a database through a login form?", True),   # SQL injection
    ("How do attackers trick employees into giving away passwords?", True),  # phishing / social eng.
    ("What stops someone tampering with evidence after it's seized?", True),  # chain of custody
]


# Hard out-of-scope: security topics that SOUND on-topic but are absent from the notes
# (checked with grep: CSRF, Bluetooth pairing, Splunk, Nmap, bug bounty). These are closer to
# the in-scope questions in embedding space than cake or football, so they test the gate harder.
HARD_OUT = [
    ("What is a CSRF attack and how is it prevented?", False),
    ("How does Bluetooth pairing security work?", False),
    ("What is Splunk used for?", False),
    ("What does Nmap do?", False),
    ("How do bug bounty programs work?", False),
]

# Follow-ups: (previous user question, follow-up). The follow-up has no topic words of its
# own, so it only passes the gate if the history rewrite works. All are in-scope.
FOLLOWUPS = [
    ("What is a firewall?", "Can you explain that more simply?"),
    ("What is phishing?", "Give me an example."),
    ("What is a DDoS attack?", "How can it be prevented?"),
]

# Offline check (no embeddings): does the rewrite fire when it should, and ONLY then?
REWRITE_CASES = [
    ("What is a firewall?", "Can you explain that more simply?", True),
    ("What is a firewall?", "Why?", True),
    ("What is a DDoS attack?", "How can it be prevented?", True),
    ("What is a firewall?", "What is phishing?", False),            # new topic: keep as is
    ("What is a firewall?", "What is phishing and how does it work?", False),  # 'it' is not a cue
    ("What is a firewall?", "How does a stateful firewall differ from a packet filter?", False),
]


def check_rewrite() -> None:
    print("\n=== rewrite rule (offline) ===")
    bad = 0
    for prev, q, expect in REWRITE_CASES:
        hist = [{"role": "user", "content": prev}, {"role": "assistant", "content": "..."}]
        fired = llm_answering.retrieval_query(q, hist) != q
        bad += fired != expect
        print(f"{'ok ' if fired == expect else 'NO '} rewritten={str(fired):<5} {q}")
    print(f"Rewrite rule correct: {len(REWRITE_CASES) - bad}/{len(REWRITE_CASES)}")


# Second stress set (vague student phrasing). It was written when the threshold was 0.80 and one
# question landed at 0.7986, which is why the default moved to 0.85. So it justified the move; it
# does NOT validate 0.85. Every topic exists in the notes (checked with grep).
STRESS2 = [
    ("What can go wrong if people reuse the same password everywhere?", True),
    ("How can you tell if an email is trying to trick you?", True),
    ("Why do companies make staff do security training?", True),
    ("What happens to my data when it travels across a network?", True),
]


def best_distance(question: str, history: list[dict] | None = None) -> float:
    dists = [c["distance"] for c in get_chunks(question, TOP_K, history) if "distance" in c]
    return min(dists) if dists else float("inf")


def run_followups() -> None:
    print(f"\n=== FOLLOW-UP set (threshold {THRESHOLD}) ===")
    print(f"{'no history':>10} {'with history':>12}  follow-up")
    correct = 0
    for prev, q in FOLLOWUPS:
        hist = [{"role": "user", "content": prev}, {"role": "assistant", "content": "..."}]
        d0, d1 = best_distance(q), best_distance(q, hist)
        correct += d1 <= THRESHOLD
        print(f"{d0:10.4f} {d1:12.4f}  {q}   (after: {prev})")
    print(f"Follow-ups passing the gate with history: {correct}/{len(FOLLOWUPS)}")


def run(name: str, questions) -> None:
    print(f"\n=== {name} set (threshold {THRESHOLD}) ===")
    print(f"{'best dist':>9}  {'expected':<9} {'gate says':<10} ok?  question")
    inside, outside, correct = [], [], 0
    for q, expected_in in questions:
        d = best_distance(q)
        passes = d <= THRESHOLD
        ok = passes == expected_in
        correct += ok
        (inside if expected_in else outside).append(d)
        print(f"{d:9.4f}  {'in' if expected_in else 'out':<9} "
              f"{'answer' if passes else 'decline':<10} {'yes' if ok else 'NO '}  {q}")
    print(f"Gate correct: {correct}/{len(questions)}")
    if inside and outside:
        print(f"Furthest in-scope: {max(inside):.4f} | closest out-of-scope: {min(outside):.4f} "
              f"| gap: {min(outside) - max(inside):+.4f}  (negative = the two groups overlap)")


if __name__ == "__main__":
    run("tuning (16 original questions)", TEST_QUESTIONS)
    run("VALIDATION (unseen questions)", VALIDATION)
    run("STRESS (student-style phrasing, in-scope only)", STRESS)
    run("STRESS 2 (vague questions; the 0.7986 one set the threshold)", STRESS2)
    run("HARD out-of-scope (on-topic-sounding, absent from notes)", HARD_OUT)
    check_rewrite()
    run_followups()