import re


def preprocessing(text: str) -> str:
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^a-zа-я]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
