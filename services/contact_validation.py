"""Utilities for validating consultation contact details."""

from __future__ import annotations

import re

__all__ = ["ContactValidationError", "validate_contact"]


class ContactValidationError(ValueError):
    """Raised when a provided contact cannot be validated."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TELEGRAM_RE = re.compile(
    r"^(?:https?://)?(?:t(?:elegram)?\.me/)?@?([A-Za-z][A-Za-z0-9_]{4,31})$"
)
_PHONE_ALLOWED_CHARS_RE = re.compile(r"[\s().-]")
_INVALID_CONTACT_MESSAGE = (
    "Не удалось распознать контакт. Укажите телефон, email или ник в Telegram,"
    " например, +79991234567, user@example.com или @username."
)


def _normalize_phone(text: str) -> str | None:
    cleaned = _PHONE_ALLOWED_CHARS_RE.sub("", text)
    if not cleaned:
        return None

    has_plus = cleaned.startswith("+")
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 10:
        return None

    if has_plus:
        return "+" + digits
    return digits


def _normalize_telegram(text: str) -> str | None:
    match = _TELEGRAM_RE.fullmatch(text)
    if not match:
        return None
    username = match.group(1)
    return f"@{username}"


def validate_contact(value: str) -> str:
    """Validate consultation contact details.

    Returns a cleaned version of the contact or raises :class:`ContactValidationError`.
    """

    text = value.strip()
    if not text:
        raise ContactValidationError(_INVALID_CONTACT_MESSAGE)

    if _EMAIL_RE.fullmatch(text):
        return text

    telegram_handle = _normalize_telegram(text)
    if telegram_handle:
        return telegram_handle

    phone = _normalize_phone(text)
    if phone:
        return phone

    raise ContactValidationError(_INVALID_CONTACT_MESSAGE)
