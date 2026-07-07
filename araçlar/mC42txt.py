import argparse
import gzip
import json
import logging
from pathlib import Path


def is_language_path(file_path: Path, lang: str) -> bool:
    lower_path = str(file_path).lower()
    lang_token = lang.lower()
    return (
        f"/{lang_token}/" in lower_path
        or f"\\{lang_token}\\" in lower_path
        or lang_token in file_path.name.lower()
    )


def open_text_file(path: Path):
    if path.suffix.lower() == ".gz":
        with path.open("rb") as raw:
            magic = raw.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(path, mode="rt", encoding="utf-8", errors="replace")
        logging.warning("File %s is not gzipped; reading as plain text.", path)
        return path.open(mode="r", encoding="utf-8", errors="replace")
    return path.open(mode="r", encoding="utf-8", errors="replace")


def iter_json_records(path: Path):
    with open_text_file(path) as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping invalid JSON line %d in %s: %s", line_number, path, exc)


def record_matches_language(record: dict, lang: str) -> bool:
    if not isinstance(record, dict):
        return False

    lang_lower = lang.lower()
    for field in ["lang", "language", "language_code", "lang_code"]:
        value = record.get(field)
        if isinstance(value, str) and value.lower().startswith(lang_lower):
            return True

    if "metadata" in record and isinstance(record["metadata"], dict):
        for field in ["lang", "language", "language_code", "lang_code"]:
            value = record["metadata"].get(field)
            if isinstance(value, str) and value.lower().startswith(lang_lower):
                return True

    return False


def extract_text_from_record(record: dict) -> str:
    if not isinstance(record, dict):
        return ""

    for field in ["text", "content", "source_text"]:
        value = record.get(field)
        if isinstance(value, str):
            return value

    return ""


def should_process_file(file_path: Path, lang: str) -> bool:
    file_name = file_path.name.lower()
    if not file_name.startswith("c4-tr.tfrecord"):
        return False
    if file_path.suffix.lower() in {".json", ".jsonl", ".gz"}:
        return True
    return False


def convert_file(input_file: Path, output_file: Path, lang: str, match_lang_per_record: bool) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open_text_file(input_file) as infile, output_file.open("w", encoding="utf-8") as outfile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSON line %d in %s", line_number, input_file)
                continue

            if match_lang_per_record and not record_matches_language(record, lang):
                continue

            text = extract_text_from_record(record)
            if text:
                outfile.write(text)
                outfile.write("\n")
                written += 1
    return written


def convert_mc4_to_txt(input_root: Path, output_root: Path, lang: str = "tr", skip_errors: bool = False) -> int:
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input path does not exist or is not a directory: {input_root}")

    converted_files = 0
    total_records = 0
    for input_file in input_root.rglob("*"):
        if input_file.is_dir():
            continue
        if not should_process_file(input_file, lang):
            continue

        relative_output = input_file.relative_to(input_root)
        if input_file.suffix.lower() == ".gz" and input_file.stem.endswith(".jsonl"):
            relative_output = relative_output.with_suffix("")
        relative_output = relative_output.with_suffix(".txt")

        output_file = output_root / relative_output
        match_lang_per_record = not is_language_path(input_file, lang)
        try:
            logging.info("Processing %s -> %s", input_file, output_file)
            written = convert_file(input_file, output_file, lang, match_lang_per_record)
            if written > 0:
                converted_files += 1
                total_records += written
            else:
                logging.info("No matching records found in %s", input_file)
        except Exception as exc:
            logging.error("Failed converting %s: %s", input_file, exc)
            if not skip_errors:
                raise

    logging.info("Wrote %d TXT files with %d matching records.", converted_files, total_records)
    return converted_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face mC4 Turkish data files to TXT by extracting the text field."
    )
    parser.add_argument("input_path", type=Path, help="Source folder containing mC4 JSON/JSONL files.")
    parser.add_argument("output_path", type=Path, help="Target folder for generated TXT files.")
    parser.add_argument("--lang", default="tr", help="Language code to filter. Defaults to tr.")
    parser.add_argument("--skip-errors", action="store_true", help="Continue processing other files if one file fails.")
    parser.add_argument("--quiet", action="store_true", help="Reduce output to warnings and errors only.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        converted = convert_mc4_to_txt(args.input_path, args.output_path, lang=args.lang, skip_errors=args.skip_errors)
    except Exception as exc:
        logging.error("Conversion failed: %s", exc)
        return 1

    logging.info("Completed conversion of %d files.", converted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
