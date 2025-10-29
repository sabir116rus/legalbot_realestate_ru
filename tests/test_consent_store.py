import asyncio
import json

import pytest

from services.consent_store import ConsentStore


@pytest.mark.asyncio
async def test_load_consents_reads_existing_values(tmp_path):
    path = tmp_path / "consents.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    store = ConsentStore(path)

    result = await store.load_consents()

    assert result == {1, 2, 3}


@pytest.mark.asyncio
async def test_add_and_remove_consent(tmp_path):
    path = tmp_path / "consents.json"
    store = ConsentStore(path)

    await store.add_consent(10)
    await store.add_consent(20)
    await store.add_consent(10)

    assert await store.load_consents() == {10, 20}

    await store.remove_consent(10)
    await store.remove_consent(99)

    assert await store.load_consents() == {20}


@pytest.mark.asyncio
async def test_concurrent_updates_are_serialized(tmp_path):
    path = tmp_path / "consents.json"
    store = ConsentStore(path)

    await asyncio.gather(*[store.add_consent(user_id) for user_id in range(50)])

    after_add = await store.load_consents()
    assert after_add == set(range(50))

    await asyncio.gather(*[store.remove_consent(user_id) for user_id in range(20)])

    after_remove = await store.load_consents()
    assert after_remove == set(range(20, 50))
