#!/usr/bin/env python3
"""Export changed draw.io sources directly to the paper's PDF figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

FIGURES = Path(__file__).resolve().parents[1] / 'figures'


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Export all figures.')
    args = parser.parse_args()
    manifest_path = FIGURES / 'figure_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    changed = []
    for figure in manifest['figures']:
        source, pdf = FIGURES / figure['source'], FIGURES / figure['pdf']
        if not source.is_file():
            raise SystemExit(f'Missing draw.io source: {source}')
        if (args.force or digest(source) != figure.get('source_sha256')
                or digest(pdf) != figure.get('pdf_sha256') or not pdf.is_file()):
            changed.append((figure, source, pdf))
    if not changed:
        print('Figure PDFs are up to date.')
        return
    executable = shutil.which(os.environ.get('DRAWIO', 'drawio'))
    if executable is None:
        raise SystemExit('Install draw.io Desktop or set DRAWIO to its executable path; '
                         'changed .drawio sources must be exported before compiling.')
    command = [executable]
    if os.name == 'posix' and not os.environ.get('DISPLAY'):
        command += ['--headless', '--disable-gpu', '--ozone-platform=headless']
    # Only for explicitly configured containers without Chromium sandbox support.
    if os.environ.get('DRAWIO_NO_SANDBOX') == '1':
        command.append('--no-sandbox')
    for figure, source, pdf in changed:
        # A failed export must never replace the last usable figure.
        with tempfile.TemporaryDirectory(prefix='drawio-', dir=FIGURES) as temp:
            output = Path(temp) / pdf.name
            result = subprocess.run(command + ['--export', '--format', 'pdf', '--crop',
                                    '--border', '2', '--output', str(output), str(source)],
                                    capture_output=True, text=True, timeout=120)
            if (result.returncode or not output.is_file()
                    or not output.read_bytes().startswith(b'%PDF-')):
                raise SystemExit(f'PDF export failed for {source.name}:\n'
                                 f'{result.stdout}\n{result.stderr}')
            output.replace(pdf)
        figure['source_sha256'] = digest(source)
        figure['pdf_sha256'] = digest(pdf)
        print(f'{source.name} -> {pdf.name}')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')


if __name__ == '__main__':
    main()
