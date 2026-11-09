"""Python stand-ins only: no MLX imports, model loads or native calls."""

import pytest

from fractal_vlm_state_probe.qwen_representation_observation import observe_qwen_visual
from fractal_vlm_state_probe.representation_observation import observe_projector


class Lazy:
    def __array__(self, *args, **kwargs):
        raise AssertionError("observer must not materialize tensors")

    @property
    def shape(self):
        raise AssertionError("observer must not inspect tensor metadata")


class Merger:
    def __init__(self):
        self.calls = []
        self.result = Lazy()

    def __call__(self, x):
        self.calls.append(x)
        return self.result


class Vision:
    def __init__(self):
        self.merger = Merger()
        self.calls = []
        self.result = Lazy()

    def __call__(self, pixels, grid, output_hidden_states=None):
        self.calls.append((pixels, grid, output_hidden_states))
        self.merger(pixels)
        return self.result


def test_off_on_off_preserves_exact_objects_without_extra_calls():
    target, pixels, grid = Vision(), Lazy(), Lazy()
    original_tower, original_merger = Vision.__call__, Merger.__call__
    assert target(pixels, grid, output_hidden_states=False) is target.result
    with observe_qwen_visual(target) as records:
        returned = target(pixels, grid, output_hidden_states=False)
        assert returned is target.result
    assert Vision.__call__ is original_tower and Merger.__call__ is original_merger
    assert target(pixels, grid, output_hidden_states=False) is target.result
    assert len(target.calls) == len(target.merger.calls) == 3
    assert len(records["vision"]) == len(records["merger"]) == 1
    assert records["merger"][0][0] is pixels
    assert records["merger"][0][1] is target.merger.result
    assert records["vision"][0][0] is pixels
    assert records["vision"][0][1] is grid
    assert records["vision"][0][2] is returned


def test_other_instances_are_forwarded_but_not_recorded():
    target, other, pixels, grid = Vision(), Vision(), Lazy(), Lazy()
    with observe_qwen_visual(target) as records:
        assert other(pixels, grid) is other.result
        assert target(pixels, grid, output_hidden_states=False) is target.result
    assert len(records["vision"]) == len(records["merger"]) == 1
    assert len(other.calls) == len(other.merger.calls) == 1


@pytest.mark.parametrize("mode", ["missing", "duplicate", "exception", "merger_only"])
def test_restoration_on_missing_duplicate_and_exception_paths(mode):
    target, pixels, grid = Vision(), Lazy(), Lazy()
    before = Vision.__call__, Merger.__call__
    with pytest.raises((ValueError, RuntimeError)):
        with observe_qwen_visual(target):
            if mode == "exception":
                raise RuntimeError("retained synthetic failure")
            if mode == "duplicate":
                target(pixels, grid, output_hidden_states=False)
                target(pixels, grid, output_hidden_states=False)
            if mode == "merger_only":
                target.merger(pixels)
    assert (Vision.__call__, Merger.__call__) == before
    if mode == "duplicate":
        assert len(target.calls) == len(target.merger.calls) == 1


@pytest.mark.parametrize(
    "kind", ["omitted_flag", "true_flag", "extra_kwarg", "wrong_arity"]
)
def test_unregistered_target_signature_rejected_before_forward(kind):
    target, pixels, grid = Vision(), Lazy(), Lazy()
    before = Vision.__call__, Merger.__call__
    with pytest.raises(ValueError, match="signature"):
        with observe_qwen_visual(target):
            if kind == "omitted_flag":
                target(pixels, grid)
            elif kind == "true_flag":
                target(pixels, grid, output_hidden_states=True)
            elif kind == "extra_kwarg":
                target(pixels, grid, output_hidden_states=False, extra=True)
            else:
                target(pixels, output_hidden_states=False)
    assert not target.calls and not target.merger.calls
    assert (Vision.__call__, Merger.__call__) == before


def test_nested_and_mixed_observers_are_rejected_without_breaking_outer():
    target, pixels, grid = Vision(), Lazy(), Lazy()
    with observe_qwen_visual(target):
        with pytest.raises(ValueError, match="nested"):
            with observe_qwen_visual(target):
                pass
        with pytest.raises(ValueError, match="nested"):
            with observe_projector(target.merger):
                pass
        target(pixels, grid, output_hidden_states=False)
    with observe_projector(target.merger):
        with pytest.raises(ValueError, match="nested"):
            with observe_qwen_visual(target):
                pass
        target.merger(pixels)


def test_original_native_failure_is_preserved_and_both_wrappers_restored():
    class BrokenMerger:
        def __call__(self, x):
            raise RuntimeError("original merger failure")

    target = Vision()
    target.merger = BrokenMerger()
    before = Vision.__call__, BrokenMerger.__call__
    with pytest.raises(RuntimeError, match="original merger failure"):
        with observe_qwen_visual(target):
            target(Lazy(), Lazy(), output_hidden_states=False)
    assert (Vision.__call__, BrokenMerger.__call__) == before


def test_missing_merger_inside_vision_is_not_accepted():
    class SilentVision(Vision):
        def __call__(self, pixels, grid, output_hidden_states=None):
            return self.result

    target = SilentVision()
    before = SilentVision.__call__, Merger.__call__
    with pytest.raises(ValueError, match="exactly one"):
        with observe_qwen_visual(target):
            target(Lazy(), Lazy(), output_hidden_states=False)
    assert (SilentVision.__call__, Merger.__call__) == before


def test_inherited_call_is_rejected_without_changing_class_dictionary():
    class InheritedVision(Vision):
        pass

    target = InheritedVision()
    assert "__call__" not in InheritedVision.__dict__
    original = Merger.__call__
    with pytest.raises(ValueError, match="explicit target"):
        with observe_qwen_visual(target):
            pass
    assert "__call__" not in InheritedVision.__dict__
    assert Merger.__call__ is original


def test_caught_native_failure_does_not_allow_another_target_attempt():
    class FailOnceMerger:
        def __init__(self):
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first native failure")
            return Lazy()

    target = Vision()
    target.merger = FailOnceMerger()
    before = Vision.__call__, FailOnceMerger.__call__
    with pytest.raises(ValueError, match="exactly one target vision"):
        with observe_qwen_visual(target):
            with pytest.raises(RuntimeError, match="first native failure"):
                target(Lazy(), Lazy(), output_hidden_states=False)
            target(Lazy(), Lazy(), output_hidden_states=False)
    assert target.merger.calls == len(target.calls) == 1
    assert (Vision.__call__, FailOnceMerger.__call__) == before
