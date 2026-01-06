import re

def normalize_text(text: str) -> str:
    # 1) Trim overall and normalize line endings to '\n'
    s = text.strip().replace("\r\n", "\n").replace("\r", "\n")

    # 2) Collapse spaces/tabs within lines 
    s = re.sub(r"[ \t]+", " ", s)

    # 3) Collapse 3+ blank lines down to 2 to avoid long vertical gaps
    s = re.sub(r"\n{3,}", "\n\n", s)

    # 4) Dedupe consecutive identical lines 
    lines = s.split("\n")
    result = []
    prev = None
    for ln in lines:
        cur = ln.rstrip()
        if prev is None or cur != prev:
            result.append(cur)
        prev = cur

    # 5) Rejoin with '\n' 
    return "\n".join(result)