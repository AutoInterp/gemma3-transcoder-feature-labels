import pytest
import torch

from delphi.clients.client import Client, Response
from delphi.latents import ActivatingExample, Latent, LatentRecord, NonActivatingExample
from delphi.scorers import DetectionScorer, FuzzingScorer
from delphi.scorers.scorer import ScorerResult


class ConstantResponseClient(Client):
    def __init__(self, text: str):
        super().__init__(model="dummy")
        self.text = text

    async def generate(self, prompt, **kwargs):
        return Response(text=self.text)


def _activating_example() -> ActivatingExample:
    return ActivatingExample(
        tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
        activations=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        str_tokens=["a", "b", "c"],
        quantile=1,
    )


def _non_activating_example() -> NonActivatingExample:
    return NonActivatingExample(
        tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
        activations=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        str_tokens=["x", "y", "z"],
        distance=0.0,
    )


def _record() -> LatentRecord:
    return LatentRecord(
        latent=Latent(module_name="layers.0", latent_index=0),
        test=[_activating_example()],
        not_active=[_non_activating_example()],
        explanation="test explanation",
    )


@pytest.mark.asyncio
async def test_detection_scorer_async_contract_returns_scorer_result():
    scorer = DetectionScorer(
        client=ConstantResponseClient("[1]"),
        n_examples_shown=1,
        verbose=False,
    )

    result = await scorer(_record())

    assert isinstance(result, ScorerResult)
    assert result.record.explanation == "test explanation"
    assert len(result.score) > 0


def test_detection_parse_casts_binary_ints_to_bool():
    scorer = DetectionScorer(
        client=ConstantResponseClient("[0, 1]"),
        n_examples_shown=2,
        verbose=False,
    )

    predictions, probabilities = scorer._parse("[0, 1]")

    assert predictions == [False, True]
    assert probabilities == [None, None]


def test_fuzzing_call_sync_contract_and_log_prob_flag():
    scorer = FuzzingScorer(
        client=ConstantResponseClient("[1]"),
        n_examples_shown=1,
        verbose=False,
        log_prob=True,
    )

    result = scorer.call_sync(_record())

    assert scorer.log_prob is True
    assert isinstance(result, ScorerResult)
    assert len(result.score) > 0
