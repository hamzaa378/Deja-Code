"""Conservative ingestion of programming *statements* (never index submitted solutions)."""
from __future__ import annotations

import ast
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

STATEMENT_ALIASES = (
    'problem_statement', 'problemstatement', 'statement', 'description',
    'problem_description', 'problem_text', 'problemtext', 'body', 'content',
    'question', 'text', 'problem',
)
TITLE_ALIASES = ('problem_name', 'problem_title', 'title', 'name')
ID_ALIASES = ('problem_code_id', 'problem_code', 'code', 'problem_id', 'slug', 'id')
TAGS_ALIASES = ('computed_tags', 'user_tags', 'tags', 'topics', 'concepts', 'categories')
DIFF_ALIASES = ('difficulty_level', 'difficulty', 'level', 'rating')
URL_ALIASES = ('problem_url', 'url', 'link')
SKIP_FILENAMES = ('solution', 'submission', 'compiler', 'sourcecode', 'submissioncode', 'answer')
MIN_WORDS = 12


def safe_str(value: Any) -> str:
    if value is None:
        return ''
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return ''
    except (TypeError, ValueError):
        pass
    return str(value)


def clean_text(value: Any) -> str:
    raw = safe_str(value)
    if '<' in raw and re.search(r'<\s*(p|div|br|pre|h[1-6]|li|span)\b', raw, flags=re.I):
        raw = BeautifulSoup(raw, 'html.parser').get_text(' ', strip=True)
    raw = html.unescape(raw)
    raw = re.sub(r'```[\s\S]*?```', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw)
    # Sample I/O is usually large and rewards superficial token matching.
    raw = re.split(r'\b(?:sample input|example input|sample output)\s*[:\n]', raw, maxsplit=1, flags=re.I)[0]
    return raw.strip()


def _first(row: dict, aliases: tuple[str, ...], override: str | None = None) -> Any:
    if override:
        # Explicit schema mapping always takes precedence.
        return row.get(override)
    canon = {str(k).lower().replace(' ', '_').replace('-', '_'): k for k in row}
    for alias in aliases:
        if alias in canon:
            val = row.get(canon[alias])
            if val is not None and safe_str(val).strip():
                return val
    return None


def parse_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = value
    elif isinstance(value, dict):
        parts = list(value.keys())
    else:
        s = safe_str(value).strip()
        if not s:
            return []
        if s.startswith(('[', '(', '{')):
            try:
                parsed = json.loads(s)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed = ast.literal_eval(s)
                except (SyntaxError, ValueError):
                    parsed = None
            if isinstance(parsed, (list, tuple, set)):
                parts = list(parsed)
            elif isinstance(parsed, dict):
                parts = list(parsed.keys())
            else:
                parts = re.split(r'[,;|]', s)
        else:
            parts = re.split(r'[,;|]', s)
    return sorted({str(x).strip().lower() for x in parts if str(x).strip()})


def _flatten_json(obj: Any) -> list[dict]:
    """Handles flat records and CodeChef-like {difficulty:{problem_code:object}} nested JSON."""
    if isinstance(obj, list):
        return [entry for item in obj for entry in _flatten_json(item)]
    if isinstance(obj, dict):
        if any(k.lower() in STATEMENT_ALIASES for k in obj):
            return [obj]
        if isinstance(obj.get('data'), list):
            return _flatten_json(obj['data'])
        output = []
        for outer_key, sub in obj.items():
            if not isinstance(sub, (list, dict)):
                continue
            if isinstance(sub, dict) and any(k.lower() in STATEMENT_ALIASES for k in sub):
                copy = dict(sub)
                copy.setdefault('problem_code', outer_key)
                output.append(copy)
            elif isinstance(sub, dict):
                for inner_key, v in sub.items():
                    if isinstance(v, dict) and any(k.lower() in STATEMENT_ALIASES for k in v):
                        copy = dict(v)
                        copy.setdefault('problem_code', inner_key)
                        copy.setdefault('difficulty', outer_key)
                        output.append(copy)
                    # Never recurse into individual solution dictionaries.
            elif isinstance(sub, list):
                output.extend(_flatten_json(sub))
        return output
    return []


def _iter_records(file: Path):
    suffix = file.suffix.lower()
    if suffix in ('.csv', '.tsv'):
        sep = '\t' if suffix == '.tsv' else ','
        for chunk in pd.read_csv(file, sep=sep, chunksize=500, dtype=str, on_bad_lines='skip',
                                 encoding='utf-8', encoding_errors='replace'):
            yield from chunk.to_dict('records')
    elif suffix == '.jsonl':
        with file.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip():
                    yield from _flatten_json(json.loads(line))
    elif suffix == '.json':
        with file.open(encoding='utf-8-sig') as stream:
            yield from _flatten_json(json.load(stream))
    elif suffix == '.parquet':
        try:
            yield from pd.read_parquet(file).to_dict('records')
        except ImportError as exc:
            raise RuntimeError('Install pyarrow to read parquet: pip install -e .[parquet]') from exc
    elif suffix in ('.txt', '.md', '.html', '.htm'):
        # Files containing raw text are considered a statement only inside a
        # directory explicitly prepared by the user, not massive source trees.
        yield {'problem_code': file.stem, 'title': file.stem.replace('_', ' '),
               'problem_statement': file.read_text(encoding='utf-8', errors='replace')}


def load_problems(path: str | Path, *, statement_col: str | None = None,
                  id_col: str | None = None, title_col: str | None = None,
                  limit: int | None = None) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f'Dataset not found: {source}')
    files = [source] if source.is_file() else sorted(
        f for f in source.rglob('*') if f.is_file()
        and f.suffix.lower() in ('.csv', '.tsv', '.json', '.jsonl', '.parquet', '.txt', '.md', '.html', '.htm')
        and not any(bad in f.stem.lower() for bad in SKIP_FILENAMES)
    )
    if not files:
        raise ValueError('No supported dataset files found. Extract the Kaggle download first.')
    unique: dict[str, dict] = {}
    diagnostics = {'rows': 0, 'missing_statement': 0, 'short_statement': 0, 'duplicates': 0}
    for file in files:
        for row in _iter_records(file):
            diagnostics['rows'] += 1
            statement = clean_text(_first(row, STATEMENT_ALIASES, statement_col))
            if not statement:
                diagnostics['missing_statement'] += 1
                continue
            if len(statement.split()) < MIN_WORDS:
                diagnostics['short_statement'] += 1
                continue
            title = clean_text(_first(row, TITLE_ALIASES, title_col)) or 'Untitled problem'
            code = safe_str(_first(row, ID_ALIASES, id_col)).strip()
            if not code:
                code = 'auto-' + hashlib.sha256((title + '\n' + statement).encode()).hexdigest()[:12]
            difficulty = safe_str(_first(row, DIFF_ALIASES)).strip()
            tags = parse_tags(_first(row, TAGS_ALIASES))
            url = safe_str(_first(row, URL_ALIASES)).strip()
            if not url and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{2,25}', code):
                url = f'https://www.codechef.com/problems/{code}'
            doc = {'code': code, 'title': title, 'statement': statement, 'tags': tags,
                   'difficulty': difficulty, 'url': url,
                   'search_text': (title + '. ' + statement)[:14000]}
            if code in unique:
                diagnostics['duplicates'] += 1
                # Keep the longer, better described statement; never merge solutions.
                if len(doc['statement']) > len(unique[code]['statement']):
                    unique[code] = doc
            else:
                unique[code] = doc
            if limit and len(unique) >= limit:
                break
        if limit and len(unique) >= limit:
            break
    if not unique:
        raise ValueError(f'No usable problem statements. Counts: {diagnostics}. '
                         'Pass --statement-col if the source uses an unusual column name. '
                         f'Files found: {[f.name for f in files[:8]]}')
    output = pd.DataFrame(unique.values()).reset_index(drop=True)
    output.attrs['diagnostics'] = diagnostics
    return output
