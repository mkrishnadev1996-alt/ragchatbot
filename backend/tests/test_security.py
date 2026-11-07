import unittest

from utils.security import create_token, decode_token, hash_password, verify_password

S = "x" * 32


class TestPasswords(unittest.TestCase):
    def test_round_trip(self):
        stored = hash_password("correct horse")
        self.assertTrue(verify_password("correct horse", stored))
        self.assertFalse(verify_password("wrong", stored))

    def test_salted(self):
        self.assertNotEqual(hash_password("same"), hash_password("same"))

    def test_malformed_hash_is_rejected(self):
        self.assertFalse(verify_password("x", "garbage"))
        self.assertFalse(verify_password("x", "bcrypt$a$b"))


class TestTokens(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(decode_token(create_token(7, S, 5), S), 7)

    def test_wrong_secret(self):
        self.assertIsNone(decode_token(create_token(7, S, 5), S + "x"))

    def test_expired(self):
        self.assertIsNone(decode_token(create_token(7, S, -1), S))

    def test_tampered(self):
        token = create_token(7, S, 5)
        self.assertIsNone(decode_token(token[:-2] + "xx", S))
        self.assertIsNone(decode_token("not-a-token", S))


if __name__ == "__main__":
    unittest.main()
