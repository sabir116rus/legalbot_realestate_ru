import pytest

from services.contact_validation import ContactValidationError, validate_contact


@pytest.mark.parametrize(
    "value,expected",
    [
        ("+7 (999) 123-45-67", "+79991234567"),
        ("89991234567", "89991234567"),
        ("user@example.com", "user@example.com"),
        ("@valid_user", "@valid_user"),
        ("t.me/ValidUser", "@ValidUser"),
        ("https://telegram.me/valid_user", "@valid_user"),
    ],
)
def test_validate_contact_returns_clean_value(value, expected):
    assert validate_contact(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",  # empty
        "not-a-contact",  # invalid text
        "user@",  # invalid email
        "12345",  # too short for phone
        "http://example.com",  # url not telegram
    ],
)
def test_validate_contact_raises_for_invalid_values(value):
    with pytest.raises(ContactValidationError) as exc_info:
        validate_contact(value)

    message = str(exc_info.value)
    assert "телефон" in message
    assert "email" in message
    assert "Telegram" in message
