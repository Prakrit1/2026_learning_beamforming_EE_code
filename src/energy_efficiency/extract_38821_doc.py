from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

import re

import olefile

# Re-extraction of 3GPP TR 38.821 (legacy binary .doc, no antiword/wvWare/
# libreoffice/pandoc/sudo available on this machine -- see
# handoff_prompt_EE_evaluation.txt's "3GPP TR 38.821 reference document"
# section for the original method this reproduces). olefile opens the OLE
# compound file, reads the WordDocument stream, and decodes it as
# single-byte CP1252 (NOT UTF-16 -- Word compresses plain-Latin-text pieces
# to 1 byte/char, confirmed against a first UTF-16 attempt that produced
# mojibake). Printable-character runs are extracted via regex.

doc_path = Path(
    project_root_path, 'src', 'energy_efficiency',
    '38821-g20 (3).doc',
)

ole = olefile.OleFileIO(str(doc_path))
stream = ole.openstream('WordDocument')
raw = stream.read()

text = raw.decode('cp1252', errors='ignore')

# printable-run extraction: keep runs of at least 4 printable/whitespace
# chars, drop short noise between them (matches the original method)
runs = re.findall(r'[\x20-\x7E\xA0-\xFF\n\r\t]{4,}', text)
full_text = '\n'.join(runs)

out_path = Path(project_root_path, 'outputs', '38821_extracted_text.txt')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(full_text, encoding='utf-8')

print(f'Extracted {len(full_text)} chars to {out_path}')

# Lower-threshold pass (min run length 1) specifically to recover short
# TOC PAGEREF cached page-number runs (e.g. "34"), which the {4,} filter
# above silently drops since they're too short to pass it.
runs_low = re.findall(r'[\x20-\x7E\xA0-\xFF\n\r\t]{1,}', text)
full_text_low = '\n'.join(runs_low)
out_path_low = Path(project_root_path, 'outputs', '38821_extracted_text_lowthresh.txt')
out_path_low.write_text(full_text_low, encoding='utf-8')
print(f'Extracted {len(full_text_low)} chars (low-threshold) to {out_path_low}')
