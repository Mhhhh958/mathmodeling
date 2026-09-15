from pathlib import Path
import hashlib
import re
import json
import numpy as np
import pandas as pd
import scipy.io as sio

RAW = Path('data/raw')
OUT = Path('outputs/raw_audit')
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def infer_fs(path):
    s = str(path).lower()
    if '12khz' in s:
        return 12000
    if '48khz' in s:
        return 48000
    if 'target_domain' in s:
        return 32000
    return np.nan


def parse_label(name, is_target):
    if is_target:
        return 'UNKNOWN'
    n = name.upper()
    if n.startswith('OR'):
        return 'OR'
    if n.startswith('IR'):
        return 'IR'
    if n.startswith('B'):
        return 'B'
    if n.startswith('N'):
        return 'N'
    return 'UNKNOWN'


def parse_fault_size(name):
    m = re.match(r'(?:OR|IR|B)(007|014|021|028)', name.upper())
    return float(m.group(1)) / 1000 if m else np.nan


def parse_load(name):
    m = re.search(r'_(\d)(?:_|\.MAT|$)', name.upper())
    return int(m.group(1)) if m else np.nan


def parse_or_position(name):
    m = re.search(r'@(3|6|12)', name.upper())
    return m.group(1) if m else ''


def parse_rpm_from_name(name):
    m = re.search(r'(\d{4})RPM', name.upper())
    return float(m.group(1)) if m else np.nan


rows = []
signal_hash_rows = []
files = sorted(RAW.rglob('*.mat'))

for path in files:
    rel = path.relative_to(RAW)
    is_target = 'target_domain' in str(rel).lower()
    fs = infer_fs(path)
    row = {
        'relative_path': str(rel),
        'filename': path.name,
        'domain': 'target' if is_target else 'source',
        'file_sha256': sha256_file(path),
        'fs_hz': fs,
        'label': parse_label(path.name, is_target),
        'fault_size_inch': parse_fault_size(path.name),
        'load_hp': parse_load(path.name),
        'or_position': parse_or_position(path.name),
        'status': 'OK',
    }
    try:
        mat = sio.loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        row['variables'] = '|'.join(keys)

        record_ids = []
        for k in keys:
            m = re.match(r'X(\d+)', k.upper())
            if m:
                record_ids.append('X' + m.group(1))
        row['record_id'] = sorted(set(record_ids))[0] if record_ids else path.stem

        rpm_value = np.nan
        rpm_source = 'unknown'
        for k in keys:
            if 'RPM' in k.upper():
                a = np.asarray(mat[k]).ravel()
                if len(a):
                    rpm_value = float(a[0])
                    rpm_source = 'MAT'
                    break
        if not np.isfinite(rpm_value):
            rpm_name = parse_rpm_from_name(path.name)
            if np.isfinite(rpm_name):
                rpm_value = rpm_name
                rpm_source = 'filename'
        row['rpm'] = rpm_value
        row['rpm_source'] = rpm_source

        sig_vars, lengths = [], []
        nonfinite_total = 0
        constant_vars, clip_flags = [], []

        for k in keys:
            a = np.asarray(mat[k]).squeeze()
            if a.ndim != 1 or a.size <= 1000:
                continue
            x = a.astype(float)
            sig_vars.append(k)
            lengths.append(len(x))

            finite = np.isfinite(x)
            nonfinite_total += int((~finite).sum())
            xf = x[finite]
            if len(xf) == 0 or np.std(xf) < 1e-12:
                constant_vars.append(k)
            if len(xf):
                xmin, xmax = np.min(xf), np.max(xf)
                extreme_rate = np.mean((xf == xmin) | (xf == xmax))
                if extreme_rate > 0.01:
                    clip_flags.append(k)

            signal_hash_rows.append({
                'relative_path': str(rel),
                'variable': k,
                'signal_sha256': hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            })

        row['signal_vars'] = '|'.join(sig_vars)
        row['signal_count'] = len(sig_vars)
        row['lengths'] = '|'.join(map(str, lengths))
        row['min_length'] = min(lengths) if lengths else np.nan
        row['max_length'] = max(lengths) if lengths else np.nan
        row['duration_min_s'] = min(lengths) / fs if lengths and np.isfinite(fs) else np.nan
        row['duration_max_s'] = max(lengths) / fs if lengths and np.isfinite(fs) else np.nan
        row['nonfinite_count'] = nonfinite_total
        row['constant_vars'] = '|'.join(constant_vars)
        row['possible_clipping_vars'] = '|'.join(clip_flags)
    except Exception as e:
        row['status'] = 'ERROR'
        row['error'] = repr(e)
    rows.append(row)


df = pd.DataFrame(rows)
sigdf = pd.DataFrame(signal_hash_rows)
df.to_csv(OUT / 'raw_file_audit.csv', index=False, encoding='utf-8-sig')
sigdf.to_csv(OUT / 'signal_hashes.csv', index=False, encoding='utf-8-sig')

file_dups = df[df.duplicated('file_sha256', keep=False)].sort_values('file_sha256')
file_dups.to_csv(OUT / 'exact_file_duplicates.csv', index=False, encoding='utf-8-sig')

signal_dups = sigdf[sigdf.duplicated('signal_sha256', keep=False)].sort_values('signal_sha256')
signal_dups.to_csv(OUT / 'exact_signal_duplicates.csv', index=False, encoding='utf-8-sig')

label_counts = df[df['domain'] == 'source']['label'].value_counts(dropna=False).to_dict()
summary = {
    'mat_files': int(len(df)),
    'readable': int((df.status == 'OK').sum()),
    'errors': int((df.status == 'ERROR').sum()),
    'source_files': int((df.domain == 'source').sum()),
    'target_files': int((df.domain == 'target').sum()),
    'source_label_counts': {str(k): int(v) for k, v in label_counts.items()},
    'files_with_nonfinite': int((df.get('nonfinite_count', pd.Series(dtype=float)).fillna(0) > 0).sum()),
    'files_with_constant_signal': int(df.get('constant_vars', pd.Series(dtype=str)).fillna('').astype(str).str.len().gt(0).sum()),
    'files_with_possible_clipping': int(df.get('possible_clipping_vars', pd.Series(dtype=str)).fillna('').astype(str).str.len().gt(0).sum()),
    'exact_duplicate_file_rows': int(len(file_dups)),
    'exact_duplicate_signal_rows': int(len(signal_dups)),
    'rpm_source_counts': {str(k): int(v) for k, v in df['rpm_source'].value_counts(dropna=False).to_dict().items()},
    'target_unknown_labels': int(((df.domain == 'target') & (df.label == 'UNKNOWN')).sum()),
    'target_lengths_unique': sorted([int(x) for x in df[df.domain == 'target']['min_length'].dropna().unique()]),
    'source_duration_min_s': float(df[df.domain == 'source']['duration_min_s'].min()),
    'source_duration_max_s': float(df[df.domain == 'source']['duration_max_s'].max()),
    'target_duration_min_s': float(df[df.domain == 'target']['duration_min_s'].min()),
    'target_duration_max_s': float(df[df.domain == 'target']['duration_max_s'].max()),
}

meta_rows = []
for xlsx in sorted(Path('data/metadata').rglob('*.xlsx')):
    try:
        book = pd.ExcelFile(xlsx)
        for sh in book.sheet_names:
            t = pd.read_excel(xlsx, sheet_name=sh)
            meta_rows.append({
                'file': str(xlsx), 'sheet': sh, 'rows': int(len(t)), 'cols': int(t.shape[1]),
                'missing_cells': int(t.isna().sum().sum()),
                'columns': '|'.join(map(str, t.columns[:50]))
            })
    except Exception as e:
        meta_rows.append({'file': str(xlsx), 'sheet': '', 'rows': -1, 'cols': -1, 'missing_cells': -1, 'columns': '', 'error': repr(e)})
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(OUT / 'metadata_table_audit.csv', index=False, encoding='utf-8-sig')

with (OUT / 'audit_summary.txt').open('w', encoding='utf-8') as f:
    for k, v in summary.items():
        f.write(f'{k}: {v}\n')

print('=== RAW AUDIT SUMMARY ===')
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('=== METADATA TABLE AUDIT ===')
if len(meta_df):
    print(meta_df.to_string(index=False))
else:
    print('No xlsx metadata tables found')
print('OUTPUT_DIR:', OUT.resolve())

# Trigger marker: workflow added after initial script commit.
