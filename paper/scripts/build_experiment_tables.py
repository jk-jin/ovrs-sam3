#!/usr/bin/env python3
"""Small paper-only renderer. Literature scalars and native fraction rows stay distinct."""
from __future__ import annotations
import argparse
import csv
import json
import re
import statistics
from decimal import Decimal, InvalidOperation
from pathlib import Path

class DataError(ValueError):
    pass

def number(value, upper):
    if value is None or str(value).strip() == '':
        return None
    try:
        n = Decimal(str(value))
    except InvalidOperation as exc:
        raise DataError(f'Invalid number: {value!r}') from exc
    if not n.is_finite() or not Decimal(0) <= n <= Decimal(upper):
        raise DataError(f'Out-of-range/non-finite number: {value!r}')
    return n

def latex(value):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','_':r'\_', '#':r'\#','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in str(value))

def read_literature(path):
    records = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        input_rows=list(csv.DictReader(stream))
    for r in input_rows:
        rid = r['result_id']
        if not rid or rid in records:
            raise DataError(f'Duplicate/empty literature result_id: {rid}')
        if r['unit'] not in {'percent','fraction'}:
            raise DataError(f'{rid}: unit must be explicit')
        number(r['value'],100 if r['unit']=='percent' else 1)
        if r['protocol_status'] not in {'matched','reference_only','unresolved','incompatible'}:
            raise DataError(f'{rid}: invalid protocol_status')
        if r['report_type'] not in {'original_reported','reproduced_by_source','requoted_by_source'}:
            raise DataError(f'{rid}: invalid report_type')
        if r['value'].strip():
            for key in ['method_citekey','source_citekey','source_version','row_label','column_label']:
                if not r[key].strip():
                    raise DataError(f'{rid}: missing provenance {key}')
            evidence = r.get('evidence_type') or 'pdf'
            if evidence == 'pdf':
                if not r['pdf_filename'].lower().endswith('.pdf') or not re.fullmatch('[0-9a-f]{64}',r['pdf_sha256']) or not r['pdf_page'].isdigit() or int(r['pdf_page']) < 1 or not r['table_id']:
                    raise DataError(f'{rid}: invalid PDF fingerprint/page/table')
            elif evidence in {'publisher_abstract','publisher_table'}:
                domain = 'https://ieeexplore.ieee.org/' if evidence == 'publisher_abstract' else 'https://www.mdpi.com/'
                locator = 'Abstract' if evidence == 'publisher_abstract' else r['table_id']
                if not r.get('source_url','').startswith(domain) or not locator or r.get('source_locator') != locator or any(r[k] for k in ('pdf_filename','pdf_sha256','pdf_page')) or (evidence == 'publisher_abstract' and r['table_id']):
                    raise DataError(f'{rid}: publisher evidence must have its own locator, not PDF provenance')
                if evidence == 'publisher_abstract' and r['protocol_status'] != 'unresolved':
                    raise DataError(f'{rid}: abstract-only protocol remains unresolved')
            else:
                raise DataError(f'{rid}: unsupported evidence_type')
            if r['dataset']=='LoveDA' and r['protocol_status'] in {'matched','reference_only'}:
                if r['class_count']!='7' or r['include_background']!='true':
                    raise DataError(f'{rid}: LoveDA must explicitly include all 7 classes')
        records[rid]=r
    return records

def read_native(path):
    if not path.exists():
        return {}
    records={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'eval_id','method_id','variant_id','seed','training_dataset','eval_dataset','eval_split','eval_protocol_id','is_full','checkpoint_sha256','miou_fraction','per_class','source','result_path'}
        if not required.issubset(reader.fieldnames or []):
            raise DataError('Native CSV lacks actual standard long-table fields')
        for r in reader:
            rid=r['eval_id']
            if not rid or rid in records:
                raise DataError(f'Duplicate/empty native eval_id: {rid}')
            number(r['miou_fraction'],1)  # Validate even unselected illegal values.
            records[rid]=r
    return records

def native_value(selector, records, row, dataset, spec, metric='miou'):
    ids=selector.get('eval_ids',[])
    if not ids:
        raise DataError('Native selection requires explicit eval_ids')
    if len(ids)!=len(set(ids)):
        raise DataError('Duplicate eval_id in selection')
    expected=selector.get('expected_seeds')
    hashes=selector.get('checkpoint_sha256_by_seed',{})
    if not expected or len(expected)!=len(set(expected)):
        raise DataError('Predeclared unique expected_seeds required')
    for name in ['eval_protocol_id','eval_split']:
        if not selector.get(name):
            raise DataError(f'Explicit native {name} required')
    values=[];seen=[]
    for rid in ids:
        if rid not in records:
            raise DataError(f'Selected native eval_id not found: {rid}')
        r=records[rid]
        if r.get('source')!='native' or r.get('record_type')!='single_run':
            raise DataError(f'{rid}: literature summaries cannot be averaged with native seeds')
        if r.get('role')!='evaluation' or r.get('status')!='completed' or r['is_full'].lower()!='true':
            raise DataError(f'{rid}: only complete formal native evaluation is accepted')
        if '/fixtures/' in r['result_path'].replace('\\','/') or not r['result_path']:
            raise DataError(f'{rid}: fixture/missing result path is not an experiment')
        try:
            seed=int(r['seed']);done=int(r['evaluated_samples']);total=int(r['expected_samples'])
        except (ValueError,KeyError) as exc:
            raise DataError(f'{rid}: seed and sample counts must be explicit') from exc
        if seed in seen or total<=0 or done!=total:
            raise DataError(f'{rid}: duplicate seed/checkpoint or incomplete sample counts')
        h=r['checkpoint_sha256']
        if not re.fullmatch('[0-9a-f]{64}',h) or hashes.get(str(seed))!=h:
            raise DataError(f'{rid}: checkpoint hash must be explicitly selected per seed')
        if r['method_id']!=row['method_id'] or r['variant_id']!=row['variant_id']:
            raise DataError(f'{rid}: method/variant mismatch')
        if r['training_dataset'].lower()!=row.get('training_dataset',spec['training_dataset']).lower():
            raise DataError(f'{rid}: training-source mismatch')
        actual=spec['dataset_aliases'].get(r['eval_dataset'].lower(),r['eval_dataset'])
        if actual!=dataset or r['eval_protocol_id']!=selector['eval_protocol_id'] or r['eval_split']!=selector['eval_split']:
            raise DataError(f'{rid}: dataset/split/protocol mismatch')
        if dataset=='LoveDA':
            classes=json.loads(r['per_class']);names=json.loads(r.get('class_names','[]'))
            wanted=['background','building','road','water','barren','forest','agricultural']
            if names!=wanted or sorted((c['class_id'],c['class_name']) for c in classes)!=list(enumerate(wanted)):
                raise DataError(f'{rid}: LoveDA seven-class/background mismatch')
            if any(c.get('union') is None or float(c['union'])<=0 or number(c.get('iou'),1) is None for c in classes):
                raise DataError(f'{rid}: missing/zero-union LoveDA class')
        if metric == 'miou':
            raw_value = r['miou_fraction']
        else:
            class_name = metric.removeprefix('iou.')
            matches = [c for c in json.loads(r['per_class']) if c['class_name'] == class_name]
            if not metric.startswith('iou.') or len(matches) != 1:
                raise DataError(f'{rid}: missing/ambiguous per-class metric {metric}')
            raw_value = matches[0].get('iou')
        n=number(raw_value,1)
        if n is None:
            raise DataError(f'{rid}: selected native score is missing')
        values.append(float(n)*100);seen.append(seed)
    if set(seen)!=set(expected):
        raise DataError('Native selected seeds differ from predeclared seed set')
    if spec['std_ddof']!=1:
        raise DataError('This renderer uses sample standard deviation (ddof=1)')
    dec=spec['native_decimals'];mean=statistics.mean(values)
    return (f'{mean:.{dec}f}'+(rf' $\pm$ {statistics.stdev(values):.{dec}f}' if len(values)>1 else ''),mean)

def scalar_literature(r):
    n=number(r['value'],100 if r['unit']=='percent' else 1)
    if n is None:
        return '--',None
    # Preserve the number of reported decimal places; multiply fraction exactly once.
    if r['unit']=='percent':
        return r['value'],float(n)
    value=n*100
    return format(value,'f'),float(value)

def resolve(row,dataset,lit,native,spec,missing):
    rid=row.get('literature_record_ids',{}).get(dataset)
    if rid:
        if not isinstance(rid,str) or rid not in lit:
            raise DataError(f'Explicit unique literature result_id required: {rid!r}')
        r=lit[rid]
        if r['dataset']!=dataset or r['method_id']!=row['method_id'] or r['method_variant']!=row['variant_id'] or r['metric']!='miou':
            raise DataError(f'{rid}: selected literature cell identity mismatch')
        if r['protocol_status']=='unresolved' and rid in spec.get('approved_reported_result_ids',[]):
            if row.get('panel')!='reference':
                raise DataError(f'{rid}: approved reported values belong in the reference panel only')
            display,value=scalar_literature(r)
            return (display+r'$^{\dagger}$' if value is not None else '--'),value,False
        if r['protocol_status'] in {'unresolved','incompatible'}:
            missing.append({'row':row['row_id'],'dataset':dataset,'reason':r['protocol_status'],'result_id':rid})
            return '--',None,False
        if row.get('panel')!='reference' and r['protocol_status']!='matched':
            raise DataError(f'{rid}: reference-only result cannot enter direct-comparison region')
        display,value=scalar_literature(r)
        return display,value,r['protocol_status']=='matched'
    selected=row.get('native_selection',{}).get(dataset)
    if selected:
        display,value=native_value(selected,native,row,dataset,spec)
        return display,value,True
    missing.append({'row':row['row_id'],'dataset':dataset,'reason':'planned_or_not_reported'})
    return '--',None,False

def measurement(selection, native, row_id, field, missing):
    """Read an explicitly selected efficiency field from an existing standard JSON.

    No efficiency schema is guessed: json_path and display_scale must be selected
    after the actual measurement format is known. No new result format is stored.
    """
    if not selection:
        missing.append({'row':row_id,'metric':field,'reason':'planned_measurement'})
        return '--'
    rid=selection['eval_id']
    if rid not in native:raise DataError(f'Measurement eval_id not found: {rid}')
    row=native[rid]
    if row.get('source')!='native' or row.get('status')!='completed' or row.get('role')!='evaluation' or row['is_full'].lower()!='true':
        raise DataError('Efficiency must come from complete native evaluation')
    path=Path(row['result_path'])
    if not path.is_file() or '/fixtures/' in path.as_posix():raise DataError('Actual standard measurement JSON required')
    raw=json.loads(path.read_text())
    if raw['eval_id']!=rid or raw['checkpoint']['sha256']!=row['checkpoint_sha256']:
        raise DataError('Measurement JSON does not match selected result/checkpoint')
    if selection.get('checkpoint_sha256')!=row['checkpoint_sha256']:
        raise DataError('Measurement checkpoint must be explicitly selected')
    json_path=selection['json_path']
    if not json_path.startswith('efficiency.'):
        raise DataError('Measurement json_path must refer to existing efficiency payload')
    value=raw
    for part in json_path.split('.'):
        if not isinstance(value,dict) or part not in value:raise DataError('Selected efficiency field missing')
        value=value[part]
    n=number(value,10**15)
    if n is None:raise DataError('Selected efficiency measurement is null')
    scale=Decimal(str(selection['display_scale']))
    if not scale.is_finite() or scale<=0:raise DataError('Positive explicit display_scale required')
    return f'{n*scale:.2f}'


def table(caption,label,columns,rows,align,wide=False,note=''):
    env='table*' if wide else 'table'
    align=align.replace('p{','L{')
    content=[f'\\begin{{{env}}}'+('[!tbp]' if wide else '[!htbp]'),r'\centering',f'\\caption{{{caption}}}',f'\\label{{{label}}}',r'\footnotesize' if wide else r'\small',r'\setlength{\tabcolsep}{2pt}',f'\\begin{{tabularx}}{{\\linewidth}}{{{align}}}',r'\toprule',' & '.join(columns)+r'\\',r'\midrule']
    for row in rows:
        if isinstance(row,str):content.append(row)
        else:content.append(' & '.join(row)+r'\\')
    content += [r'\bottomrule',r'\end{tabularx}']
    if note:content += [r'\par\vspace{3pt}',r'\begin{minipage}{\linewidth}',r'\footnotesize '+note,r'\end{minipage}']
    content.append(f'\\end{{{env}}}')
    return '\n'.join(content)+'\n'

def validate_shared_selections(spec):
    """One preselected checkpoint per variant/seed, shared across tables/domains."""
    selected = {}
    rows = [r for r in spec['main_rows'] if r['panel'] == 'native']
    rows += [dict(p, native_selection=spec['ablation_native_selection'].get(p['variant_id'], {}))
             for p in spec['ablation_rows'] + spec['freeze_rows']]
    weights = {}
    for row in rows:
        variant = row['variant_id']
        for dataset, selector in row.get('native_selection', {}).items():
            key = (variant, dataset)
            if key in selected and selected[key] != selector:
                raise DataError('The same variant/dataset must reuse one selection across tables')
            selected[key] = selector
            signature = (sorted(selector.get('expected_seeds', [])),
                         selector.get('checkpoint_sha256_by_seed', {}))
            if variant in weights and weights[variant] != signature:
                raise DataError('A variant must share seeds/checkpoints across evaluation datasets')
            weights[variant] = signature
        selections = row.get('native_selection', {})
        isaid = [selections[k] for k in ('iSAID-15-IgnBG', 'iSAID-16-BG') if k in selections]
        if len(isaid) == 2 and isaid[0]['eval_protocol_id'] == isaid[1]['eval_protocol_id']:
            raise DataError('The two iSAID columns require distinct evaluation protocols')
    if all(v in weights for v in ('semantic_only', 'semantic_presence')):
        if weights['semantic_only'] != weights['semantic_presence']:
            raise DataError('Both original SAM3 baselines must share the pretrained checkpoint')


def comparison_cells(row, lit, native, spec, missing):
    """Keep reported protocols separate without converting or duplicating scores."""
    cells = []
    protocols = row.get('reported_protocols', {})
    for dataset in spec['datasets']:
        if dataset == 'iSAID':
            if row['panel'] == 'native':
                for key in ('iSAID-15-IgnBG', 'iSAID-16-BG'):
                    selector = row.get('native_selection', {}).get(key)
                    if selector:
                        cells.append(native_value(selector, native, row, dataset, spec)[0])
                    else:
                        cells.append('--')
                        missing.append(dict(row=row['row_id'], dataset=key, reason='planned_or_not_reported'))
                continue
            value = resolve(row, dataset, lit, native, spec, missing)[0]
            protocol = protocols.get(dataset, 'unspecified')
            if protocol == '15-IgnBG':
                cells.extend([value, '--'])
            elif protocol == '16-BG':
                cells.extend(['--', value])
            elif protocol == 'unspecified':
                cells.append(r'\multicolumn{2}{c}{' + value + (r'$^{u}$' if value != '--' else '') + '}')
            else:
                raise DataError(f'Unknown iSAID display protocol: {protocol}')
            continue
        value = resolve(row, dataset, lit, native, spec, missing)[0]
        if dataset == 'UAVid' and row['panel'] == 'reference' and value != '--':
            protocol = protocols.get(dataset, 'unspecified')
            if protocol not in (7, 8, 'unspecified'):
                raise DataError(f'Unknown UAVid display protocol: {protocol}')
            value += r'$^{' + ('u' if protocol == 'unspecified' else str(protocol)) + '}$'
        cells.append(value)
    return cells


def build(args):
    spec = json.loads(args.spec.read_text())
    if spec.get('schema_version') != 2:
        raise DataError('Expected the paper table specification (schema_version=2)')
    lit = read_literature(args.literature)
    native = read_native(args.native)
    if spec['datasets'] != ['DLRSD','iSAID','LoveDA','Potsdam','Vaihingen','UAVid','UDD5','VDD']:
        raise DataError('Dataset order must be the fixed RSKT eight-dataset set')
    validate_shared_selections(spec)
    missing = []
    outputs = {}
    headers = []
    for dataset in spec['datasets']:
        if dataset == 'iSAID':
            headers.extend([r'\shortstack{iSAID\\15 cls.\\IgnBG}', r'\shortstack{iSAID\\16 cls.\\BG}'])
        elif dataset in spec['dataset_roles']:
            headers.append(r'\shortstack{' + dataset + r'\\' + latex(spec['dataset_roles'][dataset]) + '}')
        else:
            headers.append(dataset)
    column_count = len(headers) + 2
    for layout in spec['comparison_tables']:
        body = []
        group = None
        for row in spec[layout['rows']]:
            if row['comparison_group'] != group:
                group = row['comparison_group']
                if body:
                    body.append(r'\midrule')
                body.append(r'\multicolumn{' + str(column_count) + r'}{l}{\textbf{' + latex(group) + r'}}\\')
            cells = comparison_cells(row, lit, native, spec, missing)
            name = latex(row['label']).replace('×', r'$\times$')
            if row.get('citekey'):
                name += r'~\cite{' + row['citekey'] + '}'
            name += r'$^{' + row['resource'] + '}$'
            body.append([name, latex(row['adaptation_data'])] + cells)
        outputs[layout['file']] = table(
            layout['caption'], layout['label'], ['Method', layout['data_header']] + headers,
            body, r'@{}L{0.25\linewidth}L{0.08\linewidth}*{9}{C}@{}', True, layout['note'])

    body = []
    for row in (r for r in spec['main_rows'] if r['panel'] == 'native'):
        selector = row.get('native_selection', {}).get('LoveDA')
        metrics = ['iou.' + name for name in spec['per_class_names']] + ['miou']
        if selector:
            values = [native_value(selector,native,row,'LoveDA',spec,metric)[0] for metric in metrics]
        else:
            values = ['--'] * len(metrics)
            missing.extend(dict(row=row['row_id'],dataset='LoveDA',metric=metric,reason='planned_or_not_reported')
                           for metric in metrics)
        body.append([latex(row['label']).replace('×',r'$\times$')] + values)
    outputs['loveda_per_class.tex'] = table(
        r'Per-class IoU and mIoU on LoveDA (\%).', 'tab:loveda_per_class',
        ['Method','Background','Building','Road','Water','Barren','Forest','Agricultural','mIoU'],
        body,r'@{}L{0.25\linewidth}*{8}{C}@{}',True,
        'All seven classes, including background, contribute to mIoU. '
        'Each row reuses its LoveDA evaluation and checkpoint from the main comparison. '
        'LoveDA is the development domain. -- denotes unavailable results.')

    for key, filename, label, caption in [
        ('ablation_rows','ablation_core.tex','tab:ablation_core',
         r'Ablation of prompt interaction, spatial aggregation, and supervision (mIoU, \%).'),
        ('freeze_rows','ablation_freeze.tex','tab:ablation_freeze',
         r'Effect of RemoteCLIP encoder adaptation (mIoU, \%).')]:
        if not spec[key]:
            continue
        body = []
        for p in spec[key]:
            variant = p['variant_id']
            row = dict(row_id=variant, method_id='reca_sam3', variant_id=variant,
                       native_selection=spec['ablation_native_selection'].get(variant, {}))
            values = [resolve(row,d,lit,native,spec,missing)[0] for d in spec['ablation_datasets']]
            if key == 'ablation_rows':
                parameters = measurement(spec['core_parameter_selection'].get(variant), native,
                                         'core_params_' + variant, 'trainable_parameters_m', missing)
                body.append([latex(p['name'])] + values + [parameters])
            else:
                parameters = measurement(spec['freeze_parameter_selection'].get(variant), native,
                                         'freeze_params_' + variant, 'trainable_parameters_m', missing)
                body.append([latex(p['text_encoder']), latex(p['image_encoder'])] + values + [parameters])
        if key == 'ablation_rows':
            columns = ['Variant'] + spec['ablation_datasets'] + [r'\shortstack{Trainable\\(M)}']
            align = r'@{}L{0.43\linewidth}*{4}{C}@{}'
            note = ('Each row changes one design factor relative to Full. Strict self-interaction uses a diagonal '
                    'mask over expanded prompts, including aliases. Single scale uses only a 36-by-36 grid '
                    'with 12-by-12 windows. Unshared windows keep all three scales and Softmax fusion but use '
                    'separate parameters. All rows use DLRSD adaptation and fixed model-selection and inference rules. '
                    'LoveDA is the development domain. -- denotes unavailable measurements.')
        else:
            columns = [r'\shortstack{Text\\encoder}',r'\shortstack{Image\\encoder}'] + spec['ablation_datasets'] + [r'\shortstack{Trainable\\(M)}']
            align = r'@{}L{0.17\linewidth}L{0.17\linewidth}*{4}{C}@{}'
            note = ('Attention tuning updates the same attention-related parameter subset as Full; it is not full-encoder fine-tuning. '
                    'Aggregation and SDF remain trainable. The jointly tuned row reuses the Full checkpoint and scores. '
                    'LoveDA is the development domain. -- denotes unavailable results.')
        outputs[filename] = table(caption,label,columns,body,align,True,note)

    body = []
    for row in spec['efficiency_rows']:
        variant = row['row_id']
        values = [measurement(spec.get('efficiency_selection',{}).get(variant,{}).get(field),native,
                              'cost_' + variant,field,missing)
                  for field in ['total_parameters_m','trainable_parameters_m','time_ms','peak_memory_gb']]
        body.append([latex(row['name']).replace('×',r'$\times$')] + values)
    outputs['efficiency.tex'] = table(
        'Computational cost under a common evaluation setup.', 'tab:efficiency',
        ['Method',r'\shortstack{Total\\(M)}',r'\shortstack{Trainable\\(M)}',
         r'\shortstack{Latency\\(ms)}',r'\shortstack{Peak\\memory\\(GB)}'],body,
        r'@{}L{0.36\linewidth}*{4}{C}@{}',False,
        'Measurements share hardware, precision, input size, batch size, prompt count, warm-up, synchronization, and cache policy. '
        'Semantic × Presence includes both complete semantic and instance paths; ReCA-SAM3 uses student decoding only. '
        'Parameter counts are in millions. -- denotes unavailable measurements.')
    report = dict(draft=args.draft,literature_records=len(lit),native_records=len(native),
                  missing_cells=missing,protocol_locked=spec['protocol_locked'],
                  generated_tables=list(outputs),
                  selected_literature_ids=sorted({rid for row in spec['main_rows'] for rid in row.get('literature_record_ids',{}).values()}),
                  note='Ten planned training variants and two original SAM3 baselines; no synthetic scores.')
    required = {r['row_id'] for r in spec['main_rows'] if r['panel']=='native'}
    required |= {p['variant_id'] for p in spec['ablation_rows'] + spec['freeze_rows']}
    required |= {'freeze_params_' + p['variant_id'] for p in spec['freeze_rows']}
    required |= {'core_params_' + p['variant_id'] for p in spec['ablation_rows']}
    required |= {'cost_' + p['row_id'] for p in spec['efficiency_rows']}
    if not args.draft and (any(m['row'] in required for m in missing) or not spec['protocol_locked']):
        raise DataError('Formal mode: required results/protocol lock are incomplete; use --draft for placeholders')
    args.output.mkdir(parents=True,exist_ok=True)
    for name, content in outputs.items():
        (args.output/name).write_text('% Generated from paper/data; do not hand-edit numeric cells.\n' + content,encoding='utf-8')
    (args.output/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'Generated {len(outputs)} tables; literature={len(lit)}, native={len(native)}, missing/blocked={len(missing)}; draft={args.draft}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--literature',type=Path,default=Path('paper/data/literature_results.csv'))
    ap.add_argument('--native',type=Path,default=Path('paper/generated/data/results_long.csv'))
    ap.add_argument('--spec',type=Path,default=Path('paper/data/table_spec.json'))
    ap.add_argument('--output',type=Path,default=Path('paper/tables'))
    ap.add_argument('--draft',action='store_true')
    args=ap.parse_args()
    try:build(args)
    except (DataError,ValueError,KeyError) as exc:ap.exit(2,f'Invalid/incomplete data: {exc}\n')
if __name__=='__main__':main()


