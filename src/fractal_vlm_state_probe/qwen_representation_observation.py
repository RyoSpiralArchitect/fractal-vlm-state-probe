"""Lazy-object observation scaffold; this does not qualify a native Qwen path."""

from contextlib import contextmanager


@contextmanager
def observe_qwen_visual(vision_tower):
    """Retain pre-merger and final-return objects without evaluating or copying."""
    merger = vision_tower.merger
    tower_cls, merger_cls = type(vision_tower), type(merger)
    if tower_cls is merger_cls:
        raise ValueError("distinct vision and merger classes required")
    if any("__call__" not in cls.__dict__ for cls in (tower_cls, merger_cls)):
        raise ValueError("explicit target __call__ definitions required")
    tower_call, merger_call = tower_cls.__call__, merger_cls.__call__
    for call in (tower_call, merger_call):
        if getattr(call, "_representation_observer", False) or getattr(
            call, "_qwen_visual_observer", False
        ):
            raise ValueError("nested representation observation is not allowed")
    records = {"merger": [], "vision": []}
    attempts = {"merger": 0, "vision": 0}

    def observed_merger(instance, *args, **kwargs):
        if instance is merger:
            if attempts["merger"]:
                raise ValueError("exactly one target merger call required")
            attempts["merger"] += 1
            if len(args) != 1 or kwargs:
                raise ValueError("registered Qwen merger signature differs")
        result = merger_call(instance, *args, **kwargs)
        if instance is merger:
            records["merger"].append((args[0], result))
        return result

    def observed_vision(instance, *args, **kwargs):
        if instance is vision_tower:
            if attempts["vision"]:
                raise ValueError("exactly one target vision call required")
            attempts["vision"] += 1
            if (
                len(args) != 2
                or set(kwargs) != {"output_hidden_states"}
                or kwargs["output_hidden_states"] is not False
            ):
                raise ValueError("registered Qwen vision signature differs")
        result = tower_call(instance, *args, **kwargs)
        if instance is vision_tower:
            records["vision"].append((args[0], args[1], result))
        return result

    for call in (observed_merger, observed_vision):
        call._representation_observer = True
        call._qwen_visual_observer = True
    merger_installed, tower_installed = False, False
    try:
        merger_cls.__call__ = observed_merger
        merger_installed = True
        tower_cls.__call__ = observed_vision
        tower_installed = True
        yield records
        if len(records["merger"]) != 1 or len(records["vision"]) != 1:
            raise ValueError("exactly one target vision and merger call required")
    finally:
        if tower_installed:
            tower_cls.__call__ = tower_call
        if merger_installed:
            merger_cls.__call__ = merger_call
