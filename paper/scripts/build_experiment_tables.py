#!/usr/bin/env python3
"""Small paper-only renderer. Literature scalars and native fraction rows stay distinct."""
from __future__ import annotations
import argparse
import csv
import json
import math
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
            for key in ['method_citekey','source_citekey','source_version','pdf_filename','pdf_sha256','pdf_page','table_id','row_label','column_label']:
                if not r[key].strip():
                    raise DataError(f'{rid}: missing provenance {key}')
            if not r['pdf_filename'].lower().endswith('.pdf') or not re.fullmatch('[0-9a-f]{64}',r['pdf_sha256']) or int(r['pdf_page']) < 1:
                raise DataError(f'{rid}: invalid PDF fingerprint/page')
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

def native_value(selector, records, row, dataset, spec):
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
        if r['training_dataset'].lower()!=spec['training_dataset'].lower():
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
        n=number(r['miou_fraction'],1)
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

def average(cells,members,locked):
    if not locked or any(cells[d][1] is None or not cells[d][2] for d in members):
        return '--'
    return f'{statistics.mean(cells[d][1] for d in members):.2f}'

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
    content=[f'\\begin{{{env}}}[t]',r'\centering',f'\\caption{{{caption}}}',f'\\label{{{label}}}',r'\small',r'\setlength{\tabcolsep}{3pt}',f'\\begin{{tabular}}{{{align}}}',r'\toprule',' & '.join(columns)+r'\\',r'\midrule']
    for row in rows:
        if isinstance(row,str):content.append(row)
        else:content.append(' & '.join(row)+r'\\')
    content += [r'\bottomrule',r'\end{tabular}']
    if note:content += [r'\par\vspace{3pt}',r'\begin{minipage}{'+(r'\textwidth' if wide else r'\columnwidth')+'}',r'\footnotesize '+note,r'\end{minipage}']
    content.append(f'\\end{{{env}}}')
    return '\n'.join(content)+'\n'

def build(args):
    spec=json.loads(args.spec.read_text());lit=read_literature(args.literature);native=read_native(args.native)
    if spec['datasets']!=['DLRSD','iSAID','LoveDA','Potsdam','Vaihingen','UAVid','UDD5','VDD']:
        raise DataError('Dataset order must be the fixed RSKT eight-dataset set')
    missing=[];outputs={};body=[];panel=None
    full_by_seed={}
    selections=[r.get('native_selection',{}) for r in spec['main_rows'] if r.get('variant_id')=='full' and r.get('panel')=='native']
    selections += [spec['ablation_native_selection'].get('T00',{})]
    selections += list(spec['gate_native_selection'].values())
    for selection in selections:
        for sel in selection.values():
            for rid in sel.get('eval_ids',[]):
                if rid not in native:raise DataError(f'Selected native eval_id not found: {rid}')
                r=native[rid];seed=r['seed'];h=r['checkpoint_sha256']
                if seed in full_by_seed and full_by_seed[seed]!=h:
                    raise DataError('T00 and gates must share the selected Full checkpoint per seed')
                full_by_seed[seed]=h
    for row in spec['main_rows']:
        if row['panel']!=panel:
            panel=row['panel'];n=1+len(spec['datasets'])+len(spec['display_averages'])
            body.append(r'\midrule' if body else '')
            body.append(r'\multicolumn{'+str(n)+r'}{l}{'+('本文计划结果（尚未完成）' if panel=='native' else '文献参考结果（资源或协议尚未匹配，不参与排名）')+r'}\\')
        cells={d:resolve(row,d,lit,native,spec,missing) for d in spec['datasets']}
        name=latex(row['label'])+(r'~\cite{'+row['citekey']+'}' if row.get('citekey') else '')+r'$^{'+row['resource']+'}$'
        body.append([name]+[cells[d][0] for d in spec['datasets']]+[average(cells,spec['averages'][a],spec['averages_locked']) for a in spec['display_averages']])
    outputs['main_rskt8.tex']=table(r'八数据集 mIoU（\%）框架与文献参考结果。iSAID 为本文源域；其余列为目标数据集。-- 表示未完成、未报告或口径未核实，所有数值均不排名。','tab:main_rskt8',['方法']+spec['datasets']+spec['display_averages'],body,'p{0.245\\textwidth}'+'c'*10,True,r'N：本文待测；S、E、P、R、I 为表~\ref{tab:resource_settings} 的资源/来源标记。R 的数值来自 RSKT-Seg~\cite{liExploringEfficientOpenVocabulary2026} Table~1 的 iSAID/ViT-L 区；S 的数值来自 SegEarth-OV3~\cite{liSegEarthOV3ExploringSAM2026} Table~1 或 Table~4；E 为 SegEarth-OV 本地 arXiv v2 Table~1；P 为 ProC-SAM3 本地 arXiv v2 Table~I；I 为 Pi-Seg arXiv v3 Table~1 的 iSAID/ViT-L 区。LoveDA 仅展示已确认的含背景七类成绩。平均集合与协议未锁定，均值保留 --，不复制源论文的平均。')
    for section,filename,label,caption in [('core_rows','ablation_core.tex','tab:ablation_core','核心结构消融计划。所有成绩待测；每行定义见正文，结构变体需重新训练。'),('loss_rows','ablation_losses.tex','tab:ablation_losses','监督目标消融计划。Full 为同一共享 T00；其余每行仅关闭一个损失项。')]:
        body=[]
        for p in spec[section]:
            row=dict(row_id=p['plan_id'],method_id='reca_sam3',variant_id=p['variant_id'],native_selection=spec['ablation_native_selection'].get(p['plan_id'],{}))
            vals=[resolve(row,d,lit,native,spec,missing)[0] for d in spec['ablation_datasets']]
            body.append([p['plan_id']+' '+latex(p['name'])]+(['开' if p['plan_id']!='T08' else '关','开' if p['plan_id']!='T09' else '关'] if section=='loss_rows' else [])+vals)
        cols=['变体']+(['蒸馏','抑制'] if section=='loss_rows' else [])+spec['ablation_datasets']
        outputs[filename]=table(caption,label,cols,body,'p{0.30\\textwidth}'+'c'*(len(cols)-1),True,'mIoU 单位为百分比；当前 LoveDA 为开发选择域，独立正式划分仍待确认；种子计划未锁定，不生成标准差。')
    body=[]
    for p in spec['gate_rows']:
        row=dict(row_id=p['plan_id'],method_id='reca_sam3',variant_id='full',native_selection=spec['gate_native_selection'].get(p['plan_id'],{}))
        for d,sel in row['native_selection'].items():
            for rid in sel.get('eval_ids',[]):
                r=native.get(rid,{})
                for k,enabled in [('reference_prob_threshold',p['reference_gate']=='true'),('prediction_prob_threshold',p['prediction_gate']=='true')]:
                    raw=r.get(k,'');is_off=raw.strip().lower() in {'','none','null'}
                    if enabled==is_off:raise DataError(f'{rid}: gate {k} does not match {p["plan_id"]}')
        body.append([p['plan_id'],'开' if p['reference_gate']=='true' else '关','开' if p['prediction_gate']=='true' else '关']+[resolve(row,d,lit,native,spec,missing)[0] for d in spec['ablation_datasets']]+[measurement(spec.get('gate_timing_selection',{}).get(p['plan_id']),native,p['plan_id'],'time_ms',missing)])
    # Gate evaluations must all use one explicitly selected checkpoint.
    gate_hashes={r.get('checkpoint_sha256') for p in spec['gate_native_selection'].values() for sel in p.values() for rid in sel.get('eval_ids',[]) for r in [native.get(rid,{})]}
    if len(gate_hashes)>1:raise DataError('Gate analysis must reuse the same checkpoint')
    outputs['postprocess_gates.tex']=table('推理双门四组计划。同一 checkpoint、数据和提示，不增加训练次数。','tab:postprocess_gates',['组别','Reference 门','Prediction 门']+spec['ablation_datasets']+['时间（ms）'],body,'l'+'c'*6,True,r'mIoU（\%）与端到端时间均待测。关采用代码 None 语义；开启 Reference 门需额外原始 SAM 3 解码，其开销计入总时间。时间须来自实际统一硬件测量，当前不从 mIoU 长表推断。')
    outputs['efficiency.tex']=table('计算开销计划。总参数与可训练参数分列，时间和峰值显存采用统一测量设置。','tab:efficiency',['方法',r'\shortstack{总参数\\(M)}',r'\shortstack{可训练\\(M)}',r'\shortstack{时间\\(ms)}',r'\shortstack{显存\\(GB)}'],[[name]+[measurement(spec.get('efficiency_selection',{}).get(variant,{}).get(field),native,'cost_'+variant,field,missing) for field in ['total_parameters_m','trainable_parameters_m','time_ms','peak_memory_gb']] for variant,name in [('semantic_only','SAM 3 Semantic-only'),('simple_adapter','简单 adapter'),('full','T00 Full'),('single_orientation','T07 SDF 单方向')]],'L{0.31\\columnwidth}cccc',False,'固定 GPU、精度、输入、batch、提示数、chunk、缓存状态与双门。Reference 解码成本计入端到端时间；不将异硬件文献耗时用于公平排名。')
    report={'draft':args.draft,'literature_records':len(lit),'native_records':len(native),'missing_cells':missing,'averages_locked':spec['averages_locked'],'note':'Planned cells are distinct from invalid data; no training or synthetic results.'}
    # Formal mode requires the main native and planned ablation/gate cells, never optional reference coverage.
    required={r['row_id'] for r in spec['main_rows'] if r['panel']=='native'}|{p['plan_id'] for p in spec['core_rows']+spec['loss_rows']+spec['gate_rows']}|{'cost_semantic_only','cost_simple_adapter','cost_full','cost_single_orientation'}
    if not args.draft and (any(m['row'] in required for m in missing) or not spec['averages_locked']):
        raise DataError('Formal mode: required planned results/protocol lock are incomplete; use --draft for placeholders')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,text in outputs.items():(args.output/name).write_text('% Generated from paper/data; do not hand-edit numeric cells.\n'+text,encoding='utf-8')
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
