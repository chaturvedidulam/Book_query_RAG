from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOK_PATH = PROJECT_ROOT / "data" / "frankenstein.txt"


def load_book(file_path: Path) -> str:
    """Load a UTF-8 text file and return its contents."""
    if not file_path.exists():
        raise FileNotFoundError(f"Book not found: {file_path}")

    text = file_path.read_text(encoding="utf-8-sig")
    return text


def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg's header and footer."""
    start_pattern = r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_pattern = r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"

    start_match = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    end_match = re.search(end_pattern, text, flags=re.IGNORECASE | re.DOTALL)

    if start_match is None:
        raise ValueError("Could not locate the Gutenberg start marker.")

    if end_match is None:
        raise ValueError("Could not locate the Gutenberg end marker.")

    if start_match.end() >= end_match.start():
        raise ValueError("Invalid Gutenberg document boundaries.")

    text = text[start_match.end():end_match.start()]
    return text.strip()


def inspect_headings(text: str) -> None:
    """Print lines that may identify letters, chapters, or contents."""
    heading_pattern = re.compile(
        r"^\s*(?:LETTER\s+\d+|CHAPTER\s+\d+|CONTENTS)\s*$",
        flags=re.IGNORECASE,
    )

    for line_number, line in enumerate(text.splitlines(), start=1):
        if heading_pattern.match(line):
            print(f"Line {line_number:>5}: {line.strip()}")

def extract_sections(text: str) -> list[dict]:
    """
    Remove the table of contents and split the narrative
    into letters and chapters, preserving section metadata.
    """
    lines = text.splitlines()

    heading_pattern = re.compile(
        r"^\s*(Letter|Chapter)\s+(\d+)\s*$",
        flags=re.IGNORECASE,
    )

    headings = []

    for line_index, line in enumerate(lines):
        match = heading_pattern.match(line)

        if match:
            section_type = match.group(1).lower()
            section_number = int(match.group(2))

            headings.append(
                (line_index, section_type, section_number)
            )

    # Locate the table of contents.
    contents_index = next(
        (
            i for i, line in enumerate(lines)
            if line.strip().upper() == "CONTENTS"
        ),
        None,
    )

    if contents_index is None:
        raise ValueError("Could not locate the table of contents.")

    # The first Letter 1 is in the table of contents.
    # The second Letter 1 marks the actual narrative.
    letter_one_positions = [
        line_index
        for line_index, section_type, section_number in headings
        if section_type == "letter"
        and section_number == 1
        and line_index > contents_index
    ]

    if len(letter_one_positions) < 2:
        raise ValueError(
            "Could not distinguish the table of contents "
            "from the narrative."
        )

    narrative_start = letter_one_positions[1]

    # Ignore every heading before the actual narrative.
    narrative_headings = [
        heading
        for heading in headings
        if heading[0] >= narrative_start
    ]

    sections = []

    for i, (start, section_type, number) in enumerate(
        narrative_headings
    ):
        if i + 1 < len(narrative_headings):
            end = narrative_headings[i + 1][0]
        else:
            end = len(lines)

        body = "\n".join(lines[start + 1:end]).strip()

        if not body:
            raise ValueError(
                f"Empty section: {section_type.title()} {number}"
            )

        sections.append(
            {
                "section_type": section_type,
                "section_number": number,
                "title": f"{section_type.title()} {number}",
                "text": body,
            }
        )

    # Validate the expected structure of this edition.
    expected = (
        [("letter", n) for n in range(1, 5)]
        + [("chapter", n) for n in range(1, 25)]
    )

    actual = [
        (section["section_type"], section["section_number"])
        for section in sections
    ]

    if actual != expected:
        raise ValueError(
            "Unexpected section structure.\n"
            f"Expected: {expected}\n"
            f"Found: {actual}"
        )

    return sections

def chunk_sections(sections: list[dict]) -> list[dict]:
    """Split book sections recursively while preserving metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        keep_separator=True,
    )

    all_chunks = []

    for section_index, section in enumerate(sections):
        chunks = splitter.split_text(section["text"])

        for chunk_index, chunk_text in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": (
                        f"frankenstein_{section_index:02d}_"
                        f"{chunk_index:03d}"
                    ),
                    "source": BOOK_PATH.name,
                    "section_type": section["section_type"],
                    "section_number": section["section_number"],
                    "section_title": section["title"],
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }
            )

    return all_chunks


def main() -> None:
    raw_text = load_book(BOOK_PATH)
    cleaned_text = clean_gutenberg_text(raw_text)
    sections = extract_sections(cleaned_text)
    chunks = chunk_sections(sections)

    sizes = [len(chunk["text"]) for chunk in chunks]

    print(f"Book: {BOOK_PATH.name}")
    print(f"Sections: {len(sections)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Smallest chunk: {min(sizes)} characters")
    print(f"Largest chunk: {max(sizes)} characters")
    print(f"Average chunk: {sum(sizes) / len(sizes):.1f} characters")

    print("\nFirst 3 chunks:\n")

    for chunk in chunks[:3]:
        print(
            f"ID: {chunk['chunk_id']} | "
            f"Section: {chunk['section_title']} | "
            f"Length: {len(chunk['text'])}"
        )
        print(chunk["text"])
        print("-" * 60)


if __name__ == "__main__":
    main()
