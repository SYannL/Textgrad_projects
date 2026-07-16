import unittest

from textgrad.engine import EngineLM

from adversarial_gv.recording import RecordingEngine


class FakeEngine(EngineLM):
    model_string = "fake"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return "response"

    def __call__(self, prompt, system_prompt=None, **kwargs):
        return "response"


class RecordingTests(unittest.TestCase):
    def test_records_gradient_prompt_and_response(self):
        engine = RecordingEngine(FakeEngine())
        mark = engine.mark()
        engine(
            "Give feedback to a variable.",
            system_prompt="You are the gradient (feedback) engine.",
        )
        trace = engine.since(mark)
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["kind"], "backward_gradient")
        self.assertEqual(trace[0]["prompt"], "Give feedback to a variable.")
        self.assertEqual(trace[0]["response"], "response")


if __name__ == "__main__":
    unittest.main()
