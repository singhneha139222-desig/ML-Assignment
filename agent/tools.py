from typing import Tuple

from utils.validators import validate_email, validate_name, validate_platform


def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")


def execute_lead_capture(name: str, email: str, platform: str) -> Tuple[bool, str]:
    if not validate_name(name):
        return False, "Lead capture failed: invalid name."

    if not validate_email(email):
        return False, "Lead capture failed: invalid email format."

    if not validate_platform(platform):
        return False, "Lead capture failed: platform is required."

    mock_lead_capture(name=name.strip(), email=email.strip(), platform=platform.strip())
    return True, "Lead captured successfully."
