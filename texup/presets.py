"""Пресеты качества: карта класс -> имя модели для нейро-классов."""

PRESETS: dict[str, dict[str, str]] = {
    "faithful": {
        "diffuse": "realesrgan-x4plus",
        "material": "realesrgan-x4plus",
        "ui": "realesrgan-x4plus",
    },
    "detailed": {
        "diffuse": "remacri",
        "material": "remacri",
        "ui": "remacri",
    },
}

DEFAULT_PRESET = "detailed"
