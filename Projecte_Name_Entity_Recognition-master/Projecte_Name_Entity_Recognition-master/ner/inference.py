# Este archivo carga el modelo GLiNER y lo usa para encontrar entidades en textos

from __future__ import annotations

import re
from typing import Any

from gliner import GLiNER

# Nombre del modelo que se descarga de Hugging Face
MODEL_NAME = "gliner-community/gliner_small-v2.5"


def load_gliner_model(model_name: str = MODEL_NAME, device: str | None = None) -> GLiNER:
    # Carga el modelo (la primera vez lo descarga, después ya está guardado)
    import torch

    # Si no indicamos dispositivo, usa la GPU si hay y si no, la CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return GLiNER.from_pretrained(model_name, map_location=device)


def _chunks(text: str, words_per_chunk: int = 200):
    # GLiNER corta los textos muy largos, así que los partimos en trozos de 200 palabras
    # Devuelve cada trozo junto con la posición donde empieza en el texto original

    # Busca todas las palabras del texto (con su posición)
    words = list(re.finditer(r"\S+", text))

    # Va de 200 en 200 palabras
    for i in range(0, len(words), words_per_chunk):
        # El trozo empieza en la primera palabra del grupo
        start = words[i].start()
        # Y acaba en la última palabra del grupo (o del texto, si es el final)
        end = words[min(i + words_per_chunk, len(words)) - 1].end()
        yield start, text[start:end]


def predict_entities(
    model: GLiNER, text: str, labels: list[str], threshold: float = 0.5
) -> list[dict[str, Any]]:
    # Busca las entidades de un solo texto (nombres, fechas, direcciones...)
    # threshold es la confianza mínima: por debajo de 0.5 se descarta la entidad
    entities = []

    # Analizamos el texto trozo a trozo
    for offset, chunk in _chunks(text):
        for e in model.predict_entities(chunk, labels, threshold=threshold):
            # Las posiciones salen respecto al trozo, así que sumamos
            # el desplazamiento para que sean las del texto completo
            e["start"] += offset
            e["end"] += offset
            entities.append(e)
    return entities


def predict_dataset(
    model: GLiNER, samples: list[dict[str, Any]], labels: list[str], threshold: float = 0.5
) -> list[dict[str, Any]]:
    # Aplica lo anterior a todos los textos del conjunto de datos
    responses = []

    for sample in samples:
        # Para cada texto guardamos su id, el texto, las entidades encontradas y el idioma
        responses.append(
            {
                "id": sample.get("id"),
                "text": sample["text"],
                "entities": predict_entities(model, sample["text"], labels, threshold),
                "language": sample.get("language"),
            }
        )
    return responses