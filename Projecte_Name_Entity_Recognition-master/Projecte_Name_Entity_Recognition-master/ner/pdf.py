from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    # Abre un PDF y devuelve todo su texto como un único string
    pages = []

    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            # Sacamos el texto de cada página
            pages.append(page.get_text())

    # Unimos las páginas con un salto de línea entre ellas
    return "\n".join(pages)


def extract_pdf_dataset(
    records: list[dict[str, Any]],
    pdf_dir: str | Path,
    suffix: str = "",
) -> list[dict[str, Any]]:
    # Para cada registro del JSONL busca su PDF (pdf_dir/<id><suffix>.pdf),
    # extrae el texto y lo guarda en el campo "text"
    # suffix sirve para nombres como test-0001_scanned.pdf (por defecto no añade nada)
    pdf_dir = Path(pdf_dir)
    samples = []

    for record in records:
        # El PDF se llama como el id del registro, más el sufijo si lo hay
        pdf_path = pdf_dir / f"{record['id']}{suffix}.pdf"

        if not pdf_path.exists():
            raise FileNotFoundError(f"No se encuentra el PDF: {pdf_path}")

        # Copiamos el registro y le añadimos el texto sacado del PDF
        sample = dict(record)
        sample["text"] = extract_text_from_pdf(pdf_path)
        samples.append(sample)

    return samples