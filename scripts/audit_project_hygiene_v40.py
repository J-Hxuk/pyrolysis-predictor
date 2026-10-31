"""Read-only project hygiene audit. NEVER modifies or removes any project files.
python -m scripts.audit_project_hygiene_v40 --root .
"""
from pathlib import Path
import argparse,collections,json,os

REMOVABLE_CACHE_DIRS={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}
SENSITIVE_KEEP={'data','deployment','models','tests','pyrolysis','.git','.venv'}

def scan(root:Path):
    root=root.resolve()
    output=[]; cache=[]; patches=[]; biggest=[]; totals=collections.Counter()
    # Prune heavy virtualenv and version-control trees before walking, not after.
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:]=[d for d in dirs if d not in {'.venv','.git'}]
        for name in names:
            path=Path(base)/name
            rel=path.relative_to(root)
            try:size=path.stat().st_size
            except OSError:continue
            mb=round(size/1024/1024,3)
            totals[rel.parts[0] if rel.parts else '.']+=size
            if any(part in REMOVABLE_CACHE_DIRS for part in rel.parts):cache.append(str(rel))
            if name.lower().endswith('.zip') and ('patch' in name.lower() or name.lower().startswith('biochar_v')):
                patches.append(dict(path=str(rel),mb=mb))
            if size>=5*1024**2:biggest.append(dict(path=str(rel),mb=mb))
    return dict(project=str(root),cache_count=len(cache),cache_preview=cache[:40],
      historical_patch_archives=sorted(patches,key=lambda d:d['path']),
      large_files=sorted(biggest,key=lambda d:-d['mb'])[:40],
      directory_size_mb={k:round(v/1024**2,2) for k,v in sorted(totals.items(),key=lambda kv:-kv[1])},
      safe_operations=['Move *previously applied* patch ZIPs outside project to backed-up archive, AFTER latest full tests pass.',
                       'Delete regenerable cache directories only after closing Python/Streamlit.',
                       'Keep raw DOI PDFs, data/imports, active model artifacts, code, tests, SHA-256 manifests, and newest good bundle.',
                       'Do not bulk-delete old reports containing DOI provenance; archive them compressed outside working tree.',
                       'Audit script never deletes anything.'],
      danger='Do not delete data/imports v15/v41 extracted files merely because ZIP backups were moved.' )

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path.cwd());a.add_argument('--out',type=Path)
    args=a.parse_args();res=scan(args.root)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(res,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
