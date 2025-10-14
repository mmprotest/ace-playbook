import re

FORBIDDEN_TOKENS = [r"<<<?", r">>>?", r"<\|", r"\|>"]  # common meta-instruction tokens
FORBIDDEN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"(?:curl|wget)\s+https?://",          # forbid network/tool calls in bullet text
    r"(?:(?:rm\s+-rf)|shutdown|format\s+c:)", # destructive ops
]]


def sanitize_text(s: str) -> str:
    # Normalize code fences and strip stray control chars
    s = s.replace("\r\n", "\n")
    s = re.sub(r"```+\s*", "```", s)
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
    return s


def contains_forbidden(s: str) -> bool:
    if any(re.search(tok, s) for tok in FORBIDDEN_TOKENS):
        return True
    return any(p.search(s) for p in FORBIDDEN_PATTERNS)


def validate_bullet_body(body: str) -> None:
    if len(body) > 1200:
        raise ValueError("Bullet body too long (>1200 chars).")
    if contains_forbidden(body):
        raise ValueError("Bullet body contains forbidden patterns.")
