import unittest
from unittest.mock import MagicMock, patch

import httpx
from input_guard import InputGuard


def make_response(answers):
    response = MagicMock()
    response.json.return_value = {"answers": answers}
    return response


def noul(p):
    return {"type": "noul", "noul": p}


def scores(harmful=0.01, jailbreak=0.01, out_of_scope=0.01):
    return {
        "harmful": noul(harmful),
        "jailbreak": noul(jailbreak),
        "out_of_scope": noul(out_of_scope),
    }


class TestInputGuard(unittest.TestCase):
    def setUp(self):
        patcher = patch("input_guard.httpx.post")
        self.post = patcher.start()
        self.addCleanup(patcher.stop)
        self.guard = InputGuard("key", "http://guard.test", "guard-model", threshold=0.5)

    def test_request_shape(self):
        self.post.return_value = make_response(scores())

        self.guard.is_safe("What is MCP?")

        args, kwargs = self.post.call_args
        self.assertEqual(args[0], "http://guard.test")
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer key"})
        body = kwargs["json"]
        self.assertEqual(body["model"], "guard-model")
        self.assertEqual(body["state"], "What is MCP?")
        self.assertEqual(set(body["questions"]), {"harmful", "jailbreak", "out_of_scope"})
        for question in body["questions"].values():
            self.assertEqual(question["type"], "noul")
            self.assertEqual(set(question["criteria"]), {"true", "false"})

    def test_low_scores_allow(self):
        self.post.return_value = make_response(scores())
        self.assertTrue(self.guard.is_safe("What is MCP?"))

    def test_any_score_at_threshold_blocks(self):
        for field in ("harmful", "jailbreak", "out_of_scope"):
            with self.subTest(field=field):
                self.post.return_value = make_response(scores(**{field: 0.5}))
                self.assertFalse(self.guard.is_safe("x"))

    def test_refusal_answer_fails_open(self):
        answers = scores()
        answers["harmful"] = {"type": "refusal"}
        self.post.return_value = make_response(answers)
        self.assertTrue(self.guard.is_safe("x"))

    def test_missing_answer_fails_open(self):
        self.post.return_value = make_response({"harmful": noul(0.9)})
        self.assertTrue(self.guard.is_safe("x"))

    def test_http_error_fails_open(self):
        self.post.side_effect = httpx.ConnectError("down")
        self.assertTrue(self.guard.is_safe("x"))

    def test_error_status_fails_open(self):
        response = make_response(scores())
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock()
        )
        self.post.return_value = response
        self.assertTrue(self.guard.is_safe("x"))


if __name__ == "__main__":
    unittest.main()
