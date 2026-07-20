import json
import logging
import unittest

from adversarial_gv.progress_logging import (
    ProgressJsonFormatter,
    set_log_progress,
)


class ProgressLoggingTests(unittest.TestCase):
    def test_progress_is_first_jsonl_field(self):
        set_log_progress(3, 35, 2)
        formatter = ProgressJsonFormatter()

        def make_record():
            return logging.LogRecord(
                name="textgrad",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="call",
                args=(),
                exc_info=None,
            )

        parsed = json.loads(formatter.format(make_record()))
        self.assertEqual(list(parsed)[:2], ["progress", "epoch"])
        self.assertEqual(parsed["progress"], "3/35 1/2")
        self.assertEqual(parsed["epoch"], "1/1")

        parsed_second = json.loads(formatter.format(make_record()))
        self.assertEqual(parsed_second["progress"], "3/35 2/2")

    def test_same_record_is_counted_once_across_formatters(self):
        set_log_progress(2, 35, 2)
        record = logging.LogRecord(
            name="textgrad",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="call",
            args=(),
            exc_info=None,
        )
        first = json.loads(ProgressJsonFormatter().format(record))
        second = json.loads(ProgressJsonFormatter().format(record))
        self.assertEqual(first["progress"], "2/35 1/2")
        self.assertEqual(second["progress"], "2/35 1/2")

if __name__ == "__main__":
    unittest.main()
