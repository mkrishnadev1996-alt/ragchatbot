import logging
import os
import tempfile
import unittest

from utils.logging_config import setup_logging


class TestSetupLogging(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.original_level = self.root.level
        self.original_handlers = list(self.root.handlers)
        self.addCleanup(self._restore_root)

    def _restore_root(self):
        # Remove whatever setup_logging installed, then restore the original state
        setup_logging("WARNING")
        for handler in list(self.root.handlers):
            if handler not in self.original_handlers:
                self.root.removeHandler(handler)
                handler.close()
        self.root.setLevel(self.original_level)

    def test_sets_root_level_from_name(self):
        setup_logging("debug")
        self.assertEqual(self.root.level, logging.DEBUG)

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            setup_logging("LOUD")

    def test_repeated_calls_do_not_stack_handlers(self):
        setup_logging("INFO")
        setup_logging("INFO")
        setup_logging("INFO")
        added = [h for h in self.root.handlers if h not in self.original_handlers]
        self.assertEqual(len(added), 1)

    def test_log_file_receives_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "logs", "app.log")
            setup_logging("INFO", log_path)

            logging.getLogger("test.logging_config").info("hello from test")
            setup_logging("INFO")  # detaches and closes the file handler before cleanup
            with open(log_path, encoding="utf-8") as f:
                contents = f.read()

        self.assertIn("hello from test", contents)
        self.assertIn("INFO", contents)


if __name__ == "__main__":
    unittest.main()
