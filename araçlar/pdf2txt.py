import argparse
import logging
import os
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path, engine: str = "pymupdf") -> str:
    if engine == "pymupdf":
        try:
            import fitz
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is not installed. Install it with `pip install pymupdf` or use --engine pdfminer."
            ) from exc

        text = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    if engine == "pdfminer":
        try:
            from pdfminer.high_level import extract_text
        except ImportError as exc:
            raise ImportError(
                "pdfminer.six is not installed. Install it with `pip install pdfminer.six`."
            ) from exc

        return extract_text(str(pdf_path))

    raise ValueError(f"Unsupported engine: {engine}")


def convert_pdf_to_txt(input_pdf: Path, output_txt: Path, engine: str = "pymupdf") -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    text = extract_text_from_pdf(input_pdf, engine=engine)
    output_txt.write_text(text, encoding="utf-8")


def convert_all_pdfs(input_root: Path, output_root: Path, engine: str = "pymupdf", skip_errors: bool = False) -> int:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input path does not exist or is not a directory: {input_root}")

    pdf_files = list(input_root.rglob("*.pdf"))
    if not pdf_files:
        logging.warning("No PDF files found in input path: %s", input_root)
        return 0

    converted = 0
    for pdf_path in pdf_files:
        relative_path = pdf_path.relative_to(input_root).with_suffix(".txt")
        output_path = output_root / relative_path
        try:
            logging.info("Converting %s -> %s", pdf_path, output_path)
            convert_pdf_to_txt(pdf_path, output_path, engine=engine)
            converted += 1
        except Exception as exc:
            logging.error("Failed converting %s: %s", pdf_path, exc)
            if not skip_errors:
                raise

    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively convert all PDF files under an input path to TXT files under an output path."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Source directory containing PDF files to convert.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Target directory where TXT files will be written.",
    )
    parser.add_argument(
        "--engine",
        choices=["pymupdf", "pdfminer"],
        default="pymupdf",
        help="PDF text extraction engine to use. Defaults to pymupdf.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue converting other files if one file fails.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output to warnings and errors only.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        converted = convert_all_pdfs(args.input_path, args.output_path, engine=args.engine, skip_errors=args.skip_errors)
    except Exception as exc:
        logging.error("Conversion failed: %s", exc)
        return 1

    logging.info("Converted %d PDF file(s).", converted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
