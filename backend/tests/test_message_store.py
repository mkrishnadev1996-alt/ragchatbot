from feedback_store import FeedbackStore
from message_store import MessageStore


def test_history_round_trip_per_user(tmp_path):
    db = str(tmp_path / "t.db")
    store = MessageStore(db)
    store.add_exchange(1, "s1", "q1", "a1", ["Course - Lesson 1"], "m1")
    store.add_exchange(2, "s2", "other", "private", [], "m2")
    store.add_exchange(1, "s1", "q2", "a2", [], "m3")

    msgs = store.list_messages(1)
    assert [m["content"] for m in msgs] == ["q1", "a1", "q2", "a2"]
    assert msgs[1]["sources"] == ["Course - Lesson 1"]
    assert msgs[1]["message_id"] == "m1"
    assert all(m["content"] != "private" for m in msgs)


def test_history_includes_rating_and_limit(tmp_path):
    db = str(tmp_path / "t.db")
    store = MessageStore(db)
    FeedbackStore(db).set_rating("m1", 1, "up")
    store.add_exchange(1, "s", "q1", "a1", [], "m1")
    store.add_exchange(1, "s", "q2", "a2", [], "m2")

    assert [m["rating"] for m in store.list_messages(1) if m["role"] == "assistant"] == [
        "up",
        None,
    ]
    assert [m["content"] for m in store.list_messages(1, limit=2)] == ["q2", "a2"]
