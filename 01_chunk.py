"""
Stage 1: Extraction + Cleaning + Document-Aware Chunking
Offline pipeline steps 1-3 from the lecture.
"""
import docx, re, json, os

DATA_DIR = "/home/mohamedehab/ai_itida/rag_assignment/data"
FILES = {
    "HR_Policy_Canada.docx": {"doc_type": "HR policy", "country": "Canada"},
    "HR_Policy_Egypt.docx": {"doc_type": "HR policy", "country": "Egypt"},
    "HR_Policy_United_Arab_Emirates.docx": {"doc_type": "HR policy", "country": "UAE"},
    "HR_Policy_United_States.docx": {"doc_type": "HR policy", "country": "United States"},
    "ML.docx": {"doc_type": "ML lecture", "country": None},
}

HEADING_STYLES = {"Heading 1", "Heading 2", "Heading 3"}


def is_noise_line(text):
    """Detect repeated header/footer boilerplate (cleaning step)."""
    if re.match(r"^HR Policy Manual", text):
        return True
    if text.startswith("Key records for this page"):
        return True
    return False


def extract_chunks(filepath, base_meta):
    """
    Document-aware chunking: a new chunk starts at every heading.
    We track the heading 'breadcrumb' (Section > Subsection) as metadata.
    """
    d = docx.Document(filepath)
    chunks = []
    heading_stack = []
    current_text = []
    current_heading_path = None
    footer_note = None

    def flush():
        nonlocal current_text, current_heading_path, footer_note
        body = " ".join(t.strip() for t in current_text if t.strip())
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            chunks.append({
                "section_path": current_heading_path or "Introduction",
                "text": body,
                "key_records": footer_note,
            })
        current_text = []
        footer_note = None

    for p in d.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        if is_noise_line(text):
            if text.startswith("Key records for this page"):
                footer_note = text.replace("Key records for this page: ", "")
            continue
        if re.match(r"^Section \d+ —", text):
            continue

        if p.style.name in HEADING_STYLES:
            flush()
            level = int(p.style.name[-1])
            heading_stack = heading_stack[:level - 1] + [text]
            current_heading_path = " > ".join(heading_stack)
        else:
            current_text.append(text)

    flush()

    doc_name = os.path.basename(filepath)
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "chunk_id": f"{doc_name}_{i:03d}",
            "text": c["text"],
            "metadata": {
                "source_document": doc_name,
                "section": c["section_path"],
                "key_records": c["key_records"],
                **base_meta,
            }
        })
    return out


if __name__ == "__main__":
    all_chunks = []
    for fname, meta in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        doc_chunks = extract_chunks(path, meta)
        all_chunks.extend(doc_chunks)
        print(f"{fname}: {len(doc_chunks)} chunks")

    with open("chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")