"""Minimal, robust PDF extractor for NEW/docs/*.pdf.

Tries `pdfminer.six` first, then `PyPDF2`. Writes a harvested
Experiments section to `NEW/docs/experiments_extracted.txt` and
prints a short summary to stdout.
"""

from pathlib import Path
from typing import Optional


def extract_with_pdfminer(path: str) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return None
    try:
        return extract_text(path)
    except Exception:
        return None


def extract_with_pypdf2(path: str) -> Optional[str]:
    try:
        import PyPDF2
    except Exception:
        return None
    try:
        pages = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pages.append(page.extract_text() or '')
        return '\n'.join(pages)
    except Exception:
        return None


def find_experiments_section(text: str) -> Optional[str]:
    lower = text.lower()
    idx = lower.find('experiment')
    if idx == -1:
        return None
    endings = ['conclusion', 'conclusions', 'reference', 'references', 'acknowledg', 'future work', '\n\s*\d+\.']
    tail_idx = len(text)
    for e in endings:
        j = lower.find(e, idx + 1)
        if j != -1 and j < tail_idx:
            tail_idx = j
    return text[idx:tail_idx]


def main():
    repo_root = Path(__file__).resolve().parents[2]
    docs = repo_root / 'NEW' / 'docs'
    out_path = docs / 'experiments_extracted.txt'
    if not docs.exists():
        print('NEW/docs not found')
        return
    results = []
    for pdf in sorted(docs.glob('*.pdf')):
        print('\n---', pdf.name, '---\n')
        text = extract_with_pdfminer(str(pdf))
        if text is None:
            text = extract_with_pypdf2(str(pdf))
        if text is None:
            print('Could not extract text from', pdf.name)
            continue
        sec = find_experiments_section(text)
        if sec is None:
            print('No "experiment" keyword found; saving first 2000 chars.')
            sec = text[:2000]
        results.append((pdf.name, sec))
    if results:
        with open(out_path, 'w', encoding='utf-8') as f:
            for name, block in results:
                f.write(f'--- {name} ---\n')
                f.write(block)
                f.write('\n\n')
        print('Wrote experiments excerpt to', out_path)
        print('\nPreview (first 1200 chars):\n')
        print(results[0][1][:1200])
    else:
        print('No PDFs processed.')


if __name__ == '__main__':
    main()
