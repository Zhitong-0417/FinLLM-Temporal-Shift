"""Create reviewed GitHub copies without executing research or changing originals.

Requires nbformat. Cell indices in the review plan are zero-based original indices.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import re
import symtable
import builtins
import nbformat

ROOT = Path(__file__).resolve().parents[1]

DAYS = [14, 15, 16, 17, 22, 23, 24, 25, 26, 27]
# Only individually reviewed engineering cells. Mixed scientific audits stay.
REMOVE = {
    14: {1, 33, *range(34, 41)},
    15: {1, 26},
    16: {1, 4, 5, 37, 38, 39},
    17: {48, 49},
    22: {26, 27},
    23: {0, 31, 32, 33, 34},
    24: {1, 3, 5, 22, 23},
    25: {1, 7, 9, 36, 37, 38},
    26: {1, 3, 4},
    27: {1},
}
PURPOSE = {
    14: ('Chronological Benchmark Construction', 'Freeze the temporal benchmark and historical-only training and evaluation protocols.'),
    15: ('Historical Baseline Model', 'Train and freeze the historical Dataset B baseline using the pre-specified protocol.'),
    16: ('Initial Out-of-Time Evaluation', 'Evaluate the frozen historical baseline on the six strictly future windows.'),
    17: ('Statistical Validation of Temporal Degradation', 'Quantify uncertainty and decompose the observed temporal performance differences.'),
    22: ('Historical-Only Optimization Adequacy Control', 'Assess historical optimization adequacy without future-driven tuning.'),
    23: ('Optimized Historical Model Selection', 'Complete historical optimization and freeze the selected model before future evaluation.'),
    24: ('Optimized Future Evaluation', 'Evaluate the optimized historical model on the unchanged future benchmark.'),
    25: ('Residual F3 Failure Decomposition', 'Decompose the residual F3 failure using source, class, and lexical evidence.'),
    26: ('Historical Lexical-Instability Characterization', 'Construct and freeze the historical-only lexical-instability predictor.'),
    27: ('Out-of-Time Lexical-Instability Hypothesis Test', 'Test the frozen lexical-instability hypothesis on the future population.'),
}
PATH_SETUP = '''# Project paths
def find_project_root():
    current = Path.cwd().resolve()
    for path in [current, *current.parents]:
        if (path / "README.md").exists():
            return path
    raise RuntimeError(
        "Project root not found. Run this notebook from inside the repository."
    )

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

def resolve_artifact_paths(value):
    """Rebase legacy project paths in loaded metadata; never rewrite artifacts.

    Keep all non-path values, model identifiers, and scientific settings intact.
    Support the original project name and the public repository name.
    """
    if isinstance(value, dict):
        return {key: resolve_artifact_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_artifact_paths(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\\\", "/")
        for name in ("FinGPT_Project", "FinGPT-Temporal-Generalization"):
            marker = "/" + name + "/"
            if marker in normalized and "\\n" not in normalized:
                # Only standalone filesystem references, never prose containing a path.
                prefix, relative = normalized.split(marker, 1)
                if prefix.startswith("/") or (len(prefix) > 1 and prefix[1] == ":"):
                    if not any(char.isspace() for char in relative):
                        return str(PROJECT_ROOT / relative)
    return value

def load_artifact_json(stream, *args, **kwargs):
    return resolve_artifact_paths(json.load(stream, *args, **kwargs))
'''

def source(cell):
    return ''.join(cell.get('source', []))

def lines_keep(s, *ranges):
    lines = s.splitlines(keepends=True)
    return ''.join(''.join(lines[a-1:b]) for a, b in ranges)

def drop_dict_keys(s, names):
    """Delete only specified engineering metadata entries, retaining formatting."""
    spans = []
    lines = s.splitlines(keepends=True)
    for node in ast.walk(ast.parse(s)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in names:
                    spans.append((key.lineno-1, value.end_lineno))
    for a, b in sorted(spans, reverse=True):
        del lines[a:b]
    return ''.join(lines)

def replace_nodes(s, edits):
    """Source-preserving AST spans (all edited expressions here are ASCII)."""
    lines = s.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for node, replacement in sorted(edits, key=lambda x: (x[0].lineno, x[0].col_offset), reverse=True):
        start = offsets[node.lineno-1] + len(lines[node.lineno-1].encode('utf-8')[:node.col_offset].decode('utf-8'))
        end = offsets[node.end_lineno-1] + len(lines[node.end_lineno-1].encode('utf-8')[:node.end_col_offset].decode('utf-8'))
        s = s[:start] + replacement + s[end:]
    return s

def portable(s):
    # Handle the sole IPython magic cells without changing their dependency settings.
    parseable = '\n'.join('pass' if re.match(r'^\s*(?:!(?!=)|%)', l) else l for l in s.splitlines())
    tree = ast.parse(parseable)
    edits = []
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROJECT_ROOT' for t in node.targets):
            edits.append((node, '# PROJECT_ROOT is discovered in the project-path setup cell.'))
            count += len(re.findall(r'/content/[^\s\"\']+', ast.get_source_segment(parseable, node)))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'DRIVE_ROOT', 'DAY23_DRIVE_ROOT'} for t in node.targets):
            edits.append((node, ''))
            count += len(re.findall(r'/content/', ast.get_source_segment(parseable, node)))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and '/content/' in node.value:
            # Path(...) constants are handled in a second pass below.
            pass
    s = replace_nodes(s, edits)
    parseable = '\n'.join('pass' if re.match(r'^\s*(?:!(?!=)|%)', l) else l for l in s.splitlines())
    tree = ast.parse(parseable)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Path' and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and '/FinGPT_Project' in arg.value:
                relative = arg.value.split('/FinGPT_Project', 1)[1].strip('/')
                expression = 'PROJECT_ROOT' + ''.join(' / ' + repr(part) for part in relative.split('/') if part)
                edits.append((node, '(' + expression + ')'))
                count += len(re.findall(r'/content/', ast.get_source_segment(parseable, node)))
    s = replace_nodes(s, edits)
    # Consumers may load pre-cleanup manifests whose path fields still name a Drive mount.
    s = re.sub(r'\bjson\.load\(', 'load_artifact_json(', s)
    s = re.sub(r'\bDRIVE_ROOT\b', 'PROJECT_ROOT', s)
    s = re.sub(r'\bDAY23_DRIVE_ROOT\b', 'PROJECT_ROOT', s)
    s = s.replace('PROJECT_ROOT was recovered in Cell 0A from the fresh Day 23 Drive mount.',
                  'PROJECT_ROOT is discovered in the project-path setup cell.')
    s = s.replace('normalize_colab_drive_path(', 'resolve_recorded_path(')
    return s, count

def customized(day, i, s):
    if day == 17 and i == 1:
        s = s.replace('from google.colab import drive', '').replace('drive.mount("/content/drive")', '')
        s = s.replace('# Mount Google Drive and initialize paths', '# Initialize analysis paths')
    if day == 22 and i == 1:
        s = re.sub(r'# -+\n# 2\. Mount Google Drive\n# -+\n.*?(?=# -+\n# 3\.)', '', s, flags=re.S)
        s = s.replace('ENVIRONMENT BOOTSTRAP + GOOGLE DRIVE + DEPENDENCY INTEGRITY', 'ENVIRONMENT BOOTSTRAP + DEPENDENCY INTEGRITY')
    if day == 16 and i == 9:
        start = s.index('# Normalize historical Colab mount paths')
        end = s.index('\nfrozen_adapter_recorded', start)
        s = s[:start] + ('# Resolve frozen artifact locations within this repository.\n'
                         'def resolve_recorded_path(value):\n'
                         '    return None if value is None else Path(resolve_artifact_paths(value))\n\n') + s[end:]
        s = s.replace('# Physical model artifacts under current Drive mount', '# Physical frozen model artifacts')
    if day == 24 and i == 2:
        s = s[:s.index('# Basic root checks')].rstrip('# -\n') + '\n'
    if day == 24 and i == 20:
        # Retain scientific/provenance audit and its downstream completion manifest.
        s = lines_keep(s, (1, 539), (912, 1037))
        s = drop_dict_keys(s, {'canonical_notebook'})
        s = s.replace('FINAL INTEGRITY AUDIT + CANONICAL NOTEBOOK SAVE', 'FINAL SCIENTIFIC PROVENANCE MANIFEST')
    if day == 25 and i == 39:
        s = lines_keep(s, (1, 8), (25, 262))
        s = drop_dict_keys(s, {'canonical_notebook', 'final_resave_required'})
        s = s.replace('COMPLETION CHECKPOINT', 'SCIENTIFIC PROVENANCE MANIFEST')
        # Preserve the existing helper used here, originally inside the deleted copy cell.
        p = next(p for p in (ROOT / 'notebooks').glob('day25_*.ipynb') if not p.stem.endswith('_github'))
        helper_source = source(json.loads(p.read_text(encoding='utf-8'))['cells'][38])
        helper = next(n for n in ast.parse(helper_source).body if isinstance(n, ast.FunctionDef) and n.name == 'file_sha256')
        s = 'import hashlib\n' + ast.get_source_segment(helper_source, helper) + '\n\n' + s
    if day == 26 and i == 27:
        s = '# Scientific provenance consumed by Day 27\n' + lines_keep(s, (9, 11), (264, 522))
        s = drop_dict_keys(s, {'canonical_notebook'})
    if day == 27 and i == 24:
        # This stage contains scientific interpretation gates; preserve them.
        s = lines_keep(s, (1, 418))
        s = s.replace('Final artifact audit + completion checkpoint + canonical notebook save', 'Scientific closure and provenance manifest')
        s = s.replace('import nbformat\n', '').replace('import shutil\n', '')
    return s

NOTES = {
    14: 'Cells 31–32 retained: final checkpoint includes chronological boundaries, no-future selection, leakage evidence, and the protocol summary consumed downstream.',
    15: 'Configuration, runtime, pre-training, and final gates retained: they document quantization, exact training population, checkpoint selection, and the future firewall.',
    16: 'Protocol and final artifact gates retained: they verify frozen model identity and the exact out-of-time population.',
    17: 'Cell 2 retained: the inventory supplies artifact_status consumed by the scientific input audit. Stage C–F and final audits retained: they verify metric decomposition, transition denominators, and source-standardization coverage; these are scientific consistency checks.',
    22: 'Cell 25 retained: part-1 checkpoint and resume manifest are inputs to Day23 and record historical-only model-selection state. Scheduler-continuation audit retained as optimization-adequacy evidence.',
    23: 'Cell 5 retained: restart recovery also defines downstream stage paths and completion markers; deleting it would alter the workflow. Cell 30 retained as final scientific integration. Dependency repair retained to preserve the recorded environment.',
    24: 'Cell 12 retained: the synthetic smoke-test cell defines the exact prompt builder and label parser used by inference, verifies frozen prompt identity, and supplies required provenance. Mixed final audit retained; notebook discovery/copy/hash sections removed.',
    25: 'Cell 10 retained: upstream text-source inspection evaluates exact validation-ID coverage. Cells 35 and 39 retain immutable prediction identity and scientific-firewall provenance; notebook-only sections removed from 39. The unchanged file_sha256 helper was moved from deleted copy cell 38 to its consumer cell 39.',
    26: 'Vocabulary reconstruction and association reload audits retained: they validate historical-only features and deterministic scientific values. Cell 27 retains scientific completion provenance consumed by Day27; notebook save, notebook round-trip, and repeated completion audits removed.',
    27: 'Exposure reconstruction/reload audits retained: they validate frozen feature application and row alignment. Final mixed cell retains interpretation gates and scientific provenance; canonical-save and notebook-count audits removed.',
}

def python_source(nb):
    s = '\n'.join(source(c) for c in nb['cells'] if c['cell_type'] == 'code')
    return '\n'.join('pass' if re.match(r'^\s*(?:!(?!=)|%)', line) else line for line in s.splitlines())

def missing_globals(nb):
    table = symtable.symtable(python_source(nb), '<notebook>', 'exec')
    defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    def references(t):
        result = {s.get_name() for s in t.get_symbols() if s.is_global() and s.is_referenced()}
        for child in t.get_children():
            result |= references(child)
        return result
    return references(table) - defined - set(dir(builtins)) - {'display', 'get_ipython', '__name__'}

def path_inventory(nb):
    """Inspect literal/path assignments only; never execute notebook code."""
    env = {'PROJECT_ROOT': ROOT, 'DATA_DIR': ROOT/'data', 'RESULTS_DIR': ROOT/'results', 'MODELS_DIR': ROOT/'models'}
    found = []
    def value(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.Name): return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)): return [value(n) for n in node.elts]
        if isinstance(node, ast.Dict): return {value(k): value(v) for k,v in zip(node.keys,node.values)}
        if isinstance(node, ast.Subscript): return value(node.value)[value(node.slice)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div): return value(node.left) / value(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add): return value(node.left) + value(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'Path','str'} and len(node.args)==1:
            return (Path if node.func.id=='Path' else str)(value(node.args[0]))
        raise ValueError('Not a literal path expression')
    for ci,c in enumerate(nb['cells']):
        if c['cell_type']!='code': continue
        s = '\n'.join('pass' if re.match(r'^\s*(?:!(?!=)|%)', line) else line for line in source(c).splitlines())
        tree=ast.parse(s)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                try: v=value(node.value)
                except (KeyError, ValueError, TypeError, AttributeError, IndexError): continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        env[target.id]=v
                        if isinstance(v, Path) and v.is_relative_to(ROOT):
                            found.append({'cell':ci, 'variable':target.id, 'path':v.relative_to(ROOT).as_posix(), 'exists':v.exists()})
    return found

def main():
    original_files = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    original_files = [p for p in original_files if not p.stem.endswith('_github')]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in original_files}
    records = []
    for day in DAYS:
        p, = [p for p in original_files if p.name.startswith(f'day{day}_')]
        original = json.loads(p.read_text(encoding='utf-8'))
        original_normalized = nbformat.read(p, as_version=4)
        cleaned = copy.deepcopy(original)
        cells = []
        title, purpose = PURPOSE[day]
        cells.append(nbformat.v4.new_markdown_cell(f'# Day {day} — {title}\n\n{purpose}'))
        cells.append(nbformat.v4.new_code_cell('# Imports\nimport json\nfrom pathlib import Path'))
        cells.append(nbformat.v4.new_code_cell(PATH_SETUP))
        cells.append(nbformat.v4.new_code_cell('# Reproducibility settings\n# Original seeds and deterministic settings remain at their original workflow\n# positions below, including stage-specific RNG resets. No new seed is introduced.'))
        replaced = 0
        changed = []
        retained = []
        for i, c in enumerate(original['cells']):
            if i in REMOVE[day]:
                continue
            c = copy.deepcopy(c)
            s = source(c)
            if c['cell_type'] == 'code':
                t = customized(day, i, s)
                t, n = portable(t)
                replaced += n
                if t != s:
                    changed.append(i)
                c['source'] = t
            elif i == 0 and c['cell_type'] == 'markdown':
                # Keep all original methodology and scientific framing below the concise header.
                c['source'] = '\n'.join(s.splitlines()[1:]).lstrip('\n')
            cells.append(c)
            retained.append(i)
        cleaned['cells'] = cells
        # IDs are structural metadata, with deterministic values for older nbformat notebooks.
        cleaned['nbformat_minor'] = max(cleaned.get('nbformat_minor', 0), 5)
        for i, c in enumerate(cells):
            if i < 4:
                c['id'] = f'day{day}-github-header-{i}'
            else:
                c.setdefault('id', f'day{day}-cell-{i:03d}')
        out = p.with_name(p.stem + '_github.ipynb')
        nb = nbformat.from_dict(cleaned)
        nbformat.validate(nb)
        nbformat.write(nb, out)
        reread = nbformat.read(out, as_version=4)
        nbformat.validate(reread)
        # Syntax check only. IPython package-install lines are recognized without execution.
        compile(python_source(cleaned), str(out), 'exec')
        introduced = missing_globals(cleaned) - missing_globals(original)
        assert not introduced, (day, 'New undefined global names', introduced)
        # Exercise only the standard-library path setup from root and notebook directory.
        import os
        previous_cwd = Path.cwd()
        try:
            for cwd in (ROOT, ROOT/'notebooks'):
                os.chdir(cwd)
                scope = {'Path': Path, 'json': json}
                exec(PATH_SETUP, scope)
                assert scope['PROJECT_ROOT'] == ROOT
                for prefix in ('/content/drive/MyDrive/', '/content/gdrive/MyDrive/', '/content/gdrive_day23/MyDrive/'):
                    sample = prefix + 'FinGPT_Project/results/day24/optimized_future_predictions.csv'
                    assert scope['resolve_artifact_paths'](sample) == str(ROOT/'results/day24/optimized_future_predictions.csv')
                assert scope['resolve_artifact_paths']({'seed':42, 'model':'meta-llama/Meta-Llama-3-8B'}) == {'seed':42, 'model':'meta-llama/Meta-Llama-3-8B'}
        finally:
            os.chdir(previous_cwd)
        # Verify original relative order and every retained stored output/execution count exactly.
        assert len(reread.cells) == 4 + len(retained)
        for new, old_i in zip(reread.cells[4:], retained):
            old = original['cells'][old_i]
            assert new.cell_type == old['cell_type']
            if old['cell_type'] == 'code':
                assert new.get('outputs', []) == original_normalized.cells[old_i].get('outputs', [])
                assert new.get('execution_count') == old.get('execution_count')
            assert '/content/' not in new.source
            assert 'google.colab' not in new.source
        records.append(dict(day=day, original=p.name, output=out.name,
                            before=len(original['cells']), after=len(cells),
                            removed_code=sum(original['cells'][i]['cell_type']=='code' for i in REMOVE[day]),
                            removed_markdown=sum(original['cells'][i]['cell_type']=='markdown' for i in REMOVE[day]),
                            replaced=replaced, changed=changed, retained=retained,
                            original_sha256=hashes[p], cleaned_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                            existing_missing_globals=sorted(missing_globals(original)),
                            paths=path_inventory(cleaned)))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    (ROOT / 'scripts' / 'github_cleanup_audit.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    report = ['# GitHub notebook cleanup report', '',
              'Review copies only. Original notebooks and existing data/results/models are unchanged. No training, inference, bootstrap, or research cell was executed.', '',
              '## Per-notebook audit', '',
              'Cell indices below are zero-based indices in the original notebook. Four header/setup cells are added to each copy. Path counts count executable absolute project-path references replaced, excluding deleted cells and historical outputs.', '',
              '| Notebook (cleaned copy) | Before | After | Code removed | Markdown removed | Drive mount removed | Absolute Drive paths replaced | Scientific-analysis cells modified |',
              '|---|---:|---:|---:|---:|---|---:|---|']
    for r in records:
        report.append(f'| {r["output"]} | {r["before"]} | {r["after"]} | {r["removed_code"]} | {r["removed_markdown"]} | yes | {r["replaced"]} | yes (path/loading or provenance packaging only) |')
    for r in records:
        d=r['day']
        report += ['', f'### Day {d}', '', f'Original: `{r["original"]}`.',
                   f'Removed cells: {", ".join(map(str, sorted(REMOVE[d])))}.',
                   f'Retained cells with source edits: {", ".join(map(str, r["changed"])) or "none"}.',
                   'Ambiguous/mixed cells intentionally preserved: ' + NOTES[d],
                   'Unresolved dependencies: ' + (', '.join('`'+x+'`' for x in r['existing_missing_globals']) + ' are already undefined in the original notebook; intentionally not invented during cleanup.' if r['existing_missing_globals'] else 'No undefined global names detected by the static symbol scan; this does not establish runtime or data availability.'),
                   'Path assignment inventory (including existence checks) is recorded in `scripts/github_cleanup_audit.json`.']
        if d in {22, 26}:
            report.append('Unresolved path dependency: Day22 protocol-directory split files and Day26 root-directory split files differ in bytes; both original locations are preserved (see global findings).')
        if d == 23:
            report.append('Unresolved path dependency: retained Stage C recovery and workflow cells use different capped-balanced subdirectories (see global findings).')
    report += ['', '## Global summary', '',
               '- Added: ten `notebooks/*_github.ipynb` copies, this report, a minimal root `README.md`, `scripts/clean_mainline_notebooks.py`, and `scripts/github_cleanup_audit.json`.',
               '- Existing files modified: none. Originals are awaiting review before replacement.',
               '- Skipped: all non-main-line notebooks, including any Day18–Day21 experiments; all existing data, results, models, and saved numerical artifacts.',
               '- Validation: all ten copies pass nbformat validation after writing and reloading, Python syntax checks (recognizing IPython installation commands), and a static check for newly undefined global names. Path setup works from the root and notebooks directory; legacy manifest-path resolution is checked for all three original mount variants. Retained original cells stay in relative order; all their outputs and execution counts are preserved exactly. Original notebook byte hashes are unchanged.',
               '- Scientific-analysis cells modified: yes where path resolution or scientific-provenance packaging is embedded in those cells; no changes to scientific formulas, seeds, splits, thresholds, hyperparameters, metrics, bootstrap settings, or conclusions.',
               '- Existing stored outputs may still display historical Drive paths. They are historical execution records and are deliberately preserved, including numerical results.',
               '- Legacy absolute project paths in loaded JSON are rebased in memory only. Existing artifact bytes and identity hashes remain untouched.',
               '- Header reproducibility cell documents preservation of original RNG initialization locations; moving or adding seed calls could change the experiment.',
               '', '## Static dependency findings', '',
               '- Pre-existing missing definitions are listed per notebook above (Days15, 25, and 26). These require review before claiming clean-kernel end-to-end reproducibility; no scientific logic was invented to repair them.',
               '- Day23 restart-recovery cell 5 resets Stage C paths to `results/day23/stage_c_sampling_completion/runs/capped_balanced`, while retained later workflow cells also use `results/day23/stage_c_sampling_completion/capped_balanced`. This pre-existing state-dependent layout is preserved and flagged for review.',
               '- Day22 writes optimization split artifacts under `results/day22/protocol/`; Day26 reads same-named copies directly under `results/day22/`. Both locations currently exist. This pre-existing duplicate layout is preserved; file identity comparison is reported below.',
               '- JSON consumers can read the same repository-relative artifact locations as their producers after in-memory rebasing. Existing completion manifests remain required because they carry model/prediction identities and historical-only firewalls.',
               '- Removed notebook-copy fields are engineering metadata only. Completion filenames and scientific fields consumed by later days are retained.',
               '- This repository snapshot contains no Day18–Day21 notebook files. All present non-main-line notebooks (Days01–13) were skipped.',
               '- No end-to-end execution is claimed. Original missing definitions and conditional runtime states remain unresolved rather than silently changing the finalized experiment.']
    for filename in ('historical_optimization_split_manifest.csv','historical_optimization_split_audit.json'):
        a,b = ROOT/'results/day22/protocol'/filename, ROOT/'results/day22'/filename
        match = a.exists() and b.exists() and hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
        status = 'byte-identical' if match else ('both exist but differ in bytes' if a.exists() and b.exists() else 'one or both locations missing')
        report.append(f'- Day22 duplicate `{filename}`: {status}; no copy was substituted for the other.')
    report += ['', '## Main artifact handoffs', '',
               '| Producer → consumer | Preserved repository location |', '|---|---|',
               '| Day14 → Days15/16/22 | `results/day14/` frozen benchmark, split, training, and evaluation protocol JSON/CSV files |',
               '| Day15 → Day16 | `models/day15_historical_datasetB_lora/frozen_selected_adapter/`, `results/day15/frozen_model_manifest.json` and historical validation predictions/metrics |',
               '| Day16 → Days17/24 | `results/day16/future_predictions.csv` and frozen temporal metrics/manifests |',
               '| Day22 → Day23 | `results/day22/checkpoints/day22_part1_checkpoint.json`, `day22_part1_resume_manifest.json`, and `results/day22/protocol/` |',
               '| Day23 → Day24 | `results/day23/final_configuration_freeze/`, `stage_e_lora_capacity_audit/selected_adapter/`, `final_historical_validation/`, and `final_integration/` |',
               '| Day24 → Days25/26/27 | `results/day24/optimized_future_predictions.csv`, `optimized_future_prediction_manifest.json`, and `day24_completion_checkpoint.json` |',
               '| Day25 → Day26 | `results/day25/` lexical tables, analysis manifest, and scientific completion manifest (pre-existing missing variable definitions flagged above) |',
               '| Day26 → Day27 | `results/day26/token_instability_rank.csv`, `day26_day27_handoff_manifest.json`, and `day26_completion_checkpoint.json` |']
    (ROOT / 'GITHUB_CLEANUP_REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k not in {'retained','changed','paths'}} for r in records], indent=2))

if __name__ == '__main__':
    main()
