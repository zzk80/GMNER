from __future__ import annotations

from scripts.build_siglip2_region_cache import SeparatedSiglipProcessor


class _Recorder:
    def __init__(self, key: str) -> None:
        self.key = key
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {self.key: kwargs}


def test_separated_siglip_processor_routes_text_and_images() -> None:
    tokenizer = _Recorder("text")
    image_processor = _Recorder("image")
    processor = SeparatedSiglipProcessor(tokenizer, image_processor)

    text = processor(text=["entity"], padding=True)
    image = processor(images=["crop"], return_tensors="pt")

    assert text["text"]["text"] == ["entity"]
    assert image["image"]["images"] == ["crop"]
    assert len(tokenizer.calls) == 1
    assert len(image_processor.calls) == 1
