import unittest

from textgrad.engine import EngineLM

from adversarial_gv.recording import RecordingEngine


class FakeEngine(EngineLM):
    model_string = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, system_prompt=None, **kwargs):
        return "response"

    def __call__(self, prompt, system_prompt=None, **kwargs):
        self.calls.append({"system_prompt": system_prompt, **kwargs})
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
        self.assertEqual(engine.engine.calls[0]["max_tokens"], 512)
        self.assertTrue(
            engine.engine.calls[0]["system_prompt"].endswith(
                "Return at most 500 tokens."
            )
        )

    def test_keeps_default_length_for_non_backward_calls(self):
        engine = RecordingEngine(FakeEngine())
        engine("Evaluate this answer.", system_prompt="You are an evaluator.")
        self.assertNotIn("max_tokens", engine.engine.calls[0])
        self.assertEqual(
            engine.engine.calls[0]["system_prompt"], "You are an evaluator."
        )

    def test_preserves_explicit_backward_length(self):
        engine = RecordingEngine(FakeEngine())
        engine(
            "Give feedback to a variable.",
            system_prompt="You are the gradient (feedback) engine.",
            max_tokens=321,
        )
        self.assertEqual(engine.engine.calls[0]["max_tokens"], 321)


if __name__ == "__main__":
    unittest.main()
