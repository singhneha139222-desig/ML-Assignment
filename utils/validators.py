import re
from typing import Dict, Optional


EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
NAME_REGEX = re.compile(r"^[A-Za-z][A-Za-z\s'\-]{1,79}$")

KNOWN_PLATFORMS = {
    "youtube",
    "instagram",
    "tiktok",
    "linkedin",
    "facebook",
    "x",
    "twitter",
    "shopify",
    "web",
    "ios",
    "android",
}


def validate_email(email: str) -> bool:
    return bool(email and EMAIL_REGEX.match(email.strip()))


def validate_name(name: str) -> bool:
    cleaned = (name or "").strip()
    return bool(cleaned and NAME_REGEX.match(cleaned))


def validate_platform(platform: str) -> bool:
    return bool((platform or "").strip())


def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if not match:
        return None
    return match.group(0).strip()


def extract_name(text: str) -> Optional[str]:
    patterns = [
        r"(?:my name is|i am|i'm|this is)\s+([A-Za-z][A-Za-z\s'\-]{1,79})",
        r"name\s*[:\-]\s*([A-Za-z][A-Za-z\s'\-]{1,79})",
    ]
    lowered = text.strip()
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" .,!?")
            if validate_name(candidate):
                return candidate

    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if 1 <= len(words) <= 3 and not any(ch.isdigit() for ch in text):
        candidate = " ".join(words).strip(" .,!?")
        if validate_name(candidate):
            return candidate
    return None


def extract_platform(text: str) -> Optional[str]:
    lowered = text.lower()

    for known in KNOWN_PLATFORMS:
        if re.search(rf"\b{re.escape(known)}\b", lowered):
            return known

    match = re.search(r"(?:platform|channel|on)\s*[:\-]?\s*([A-Za-z0-9_\-]{2,40})", lowered)
    if match:
        candidate = match.group(1).strip()
        if candidate not in ["my", "the", "a", "an"]:
            return candidate

    return None


def extract_entities_from_text(text: str) -> Dict[str, str]:
    entities: Dict[str, str] = {}

    name = extract_name(text)
    if name:
        entities["name"] = name

    email = extract_email(text)
    if email:
        entities["email"] = email

    platform = extract_platform(text)
    if platform:
        entities["platform"] = platform

    return entities
