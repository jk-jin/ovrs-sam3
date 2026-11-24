"""Boundary checks for the paper renderer; all synthetic records stay in memory/temp dirs."""
import csv
import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
spec=importlib.util.spec_from_file_location('renderer',Path(__file__).with_name('build_experiment_tables.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RendererBoundaries(unittest.TestCase):
    def test_approved_reported_value_preserves_protocol_limit(self):
        record=dict(dataset='LoveDA',method_id='cat',method_variant='isaid',metric='miou',protocol_status='unresolved',value='25.11',unit='percent')
        row=dict(row_id='cat',method_id='cat',variant_id='isaid',panel='reference',literature_record_ids={'LoveDA':'reported'})
        def resolve(spec):return m.resolve(row,'LoveDA',{'reported':record},{},spec,[])
        self.assertEqual(resolve({}),('--',None,False))
        approved={'approved_reported_result_ids':['reported']}
        display,value,matched=resolve(approved)
        self.assertEqual(value,25.11)
        self.assertIn('dagger',display)
        self.assertFalse(matched)
        row['panel']='native'
        with self.assertRaises(m.DataError):resolve(approved)
        row['panel']='reference';record['protocol_status']='incompatible'
        self.assertEqual(resolve(approved),('--',None,False))

    def test_zero_missing_and_units(self):
        self.assertEqual(m.scalar_literature({'value':'0','unit':'percent'}),('0',0))
        self.assertEqual(m.scalar_literature({'value':'','unit':'percent'}),('--',None))
        self.assertEqual(m.scalar_literature({'value':'0.25','unit':'fraction'}),('25.00',25))
        for n in ['NaN','inf','-0.1','100.01','--']:
            with self.assertRaises(m.DataError):m.number(n,100)
        with self.assertRaises(m.DataError):m.number('25',1)

    def test_native_conversion_and_duplicate_seed(self):
        h='a'*64
        names=['background','building','road','water','barren','forest','agricultural']
        r=dict(eval_id='e1',source='native',record_type='single_run',role='evaluation',status='completed',is_full='true',result_path='/tmp/synthetic/result.json',seed='9',evaluated_samples='2',expected_samples='2',checkpoint_sha256=h,method_id='reca_sam3',variant_id='full',training_dataset='dlrsd',eval_dataset='loveda',eval_protocol_id='p',eval_split='test',miou_fraction='0',class_names=json.dumps(names),per_class=json.dumps([dict(class_id=i,class_name=n,iou=0,union=1) for i,n in enumerate(names)]))
        selector=dict(eval_ids=['e1'],expected_seeds=[9],checkpoint_sha256_by_seed={'9':h},eval_protocol_id='p',eval_split='test')
        config=dict(training_dataset='dlrsd',dataset_aliases={'loveda':'LoveDA'},native_decimals=2,std_ddof=1)
        row=dict(method_id='reca_sam3',variant_id='full')
        def resolve():return m.native_value(selector,records,row,'LoveDA',config)
        records={'e1':r}
        self.assertEqual(resolve(),('0.00',0))
        self.assertEqual(m.native_value(selector,records,row,'LoveDA',config,'iou.road'),('0.00',0))
        with self.assertRaises(m.DataError):
            m.native_value(selector,records,row,'LoveDA',config,'iou.missing')
        r['miou_fraction']='0.25';self.assertEqual(resolve(),('25.00',25))
        records['e2']=dict(r,eval_id='e2');selector['eval_ids']=['e1','e2']
        with self.assertRaises(m.DataError):resolve()
        selector['eval_ids']=['e1']
        for field,value in [('miou_fraction','25'),('is_full','false'),('checkpoint_sha256','b'*64),('record_type','reported_summary')]:
            old=r[field];r[field]=value
            with self.assertRaises(m.DataError):resolve()
            r[field]=old
        r['class_names']=json.dumps(names[1:])
        with self.assertRaises(m.DataError):resolve()
        r['class_names']=json.dumps(names);r['per_class']=json.dumps([dict(class_id=i,class_name=n,iou=0,union=0) for i,n in enumerate(names)])
        with self.assertRaises(m.DataError):resolve()

    def test_original_sam3_has_no_adaptation_source(self):
        h='e'*64
        r=dict(eval_id='e',source='native',record_type='single_run',role='evaluation',status='completed',
               is_full='true',result_path='/tmp/synthetic/result.json',seed='9',evaluated_samples='2',
               expected_samples='2',checkpoint_sha256=h,method_id='sam3_semantic',variant_id='semantic_only',
               training_dataset='none',eval_dataset='potsdam',eval_protocol_id='p',eval_split='val',miou_fraction='0.2')
        selector=dict(eval_ids=['e'],expected_seeds=[9],checkpoint_sha256_by_seed={'9':h},eval_protocol_id='p',eval_split='val')
        config=dict(training_dataset='dlrsd',dataset_aliases={'potsdam':'Potsdam'},native_decimals=2,std_ddof=1)
        row=dict(method_id='sam3_semantic',variant_id='semantic_only',training_dataset='none')
        self.assertEqual(m.native_value(selector,{'e':r},row,'Potsdam',config),('20.00',20.0))
        r['training_dataset']='dlrsd'
        with self.assertRaises(m.DataError):m.native_value(selector,{'e':r},row,'Potsdam',config)

    def test_shared_full_checkpoint_across_tables_and_domains(self):
        paper=Path(__file__).resolve().parents[1]
        config=json.loads((paper/'data/table_spec.json').read_text())
        selector=dict(eval_ids=['a'],expected_seeds=[9],checkpoint_sha256_by_seed={'9':'a'*64},eval_protocol_id='p',eval_split='val')
        full=next(r for r in config['main_rows'] if r['row_id']=='full')
        full['native_selection']={'LoveDA':selector}
        config['ablation_rows']=[dict(variant_id='full')]
        config['ablation_native_selection']={'full':{'LoveDA':dict(selector)}}
        m.validate_shared_selections(config)
        config['ablation_native_selection']['full']['LoveDA']=dict(selector,eval_ids=['b'])
        with self.assertRaises(m.DataError):m.validate_shared_selections(config)
        config['ablation_native_selection']={}
        full['native_selection']['Potsdam']=dict(selector,checkpoint_sha256_by_seed={'9':'b'*64})
        with self.assertRaises(m.DataError):m.validate_shared_selections(config)

    def test_current_tables_and_source_audit(self):
        paper=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(spec=paper/'data/table_spec.json',literature=paper/'data/literature_results.csv',
                                 native=Path(d)/'absent.csv',output=Path(d)/'tables',draft=True)
            m.build(args)
            expected={'main_rskt8.tex','loveda_per_class.tex',
                      'ablation_core.tex','ablation_freeze.tex','efficiency.tex'}
            self.assertEqual({p.name for p in args.output.glob('*.tex')},expected)
            for p in args.output.glob('*.tex'):
                text=p.read_text()
                self.assertNotRegex(text,r'\b[TPB]\d{2}\b')
                self.assertNotIn('Generic CLIP',text)
            main=(args.output/'main_rskt8.tex').read_text()
            self.assertIn('Adaptation',main)
            self.assertIn('GSNet',main)
            self.assertIn('Training on remote sensing segmentation data',main)
            self.assertIn('Training-free',main)
            self.assertIn('GLoP-SAM3 (ROSE report)',main)
            self.assertNotIn('GFMR',main)
            self.assertNotIn('VLex',main)
            self.assertNotIn('ConceptSeg',main)
            self.assertNotIn('Target\\\\avg.',main)
            self.assertIn(r'iSAID\\15 cls.\\IgnBG',main)
            self.assertIn(r'iSAID\\16 cls.\\BG',main)
            self.assertIn(r'\multicolumn{11}',main)
            self.assertIn('same checkpoint per method/seed',main)
            core=(args.output/'ablation_core.tex').read_text()
            self.assertIn('Unshared multiscale windows',core)
            self.assertIn('Strict prompt self-interaction',core)
            self.assertNotIn('shifted',core.lower())
            per_class=(args.output/'loveda_per_class.tex').read_text()
            self.assertIn('Background',per_class)
            self.assertIn('Agricultural',per_class)
            self.assertIn('mIoU',per_class)
            args.draft=False
            with self.assertRaises(m.DataError):m.build(args)
        audit=json.loads((paper/'data/comparison_audit_20261124.json').read_text())
        config=json.loads((paper/'data/table_spec.json').read_text())
        selected={rid for r in config['main_rows'] for rid in r.get('literature_record_ids',{}).values()}
        self.assertEqual(len(selected),156)
        self.assertEqual({r['result_id'] for r in audit['records']},selected)
        sources=m.read_literature(paper/'data/literature_results.csv')
        for r in audit['records']:
            self.assertEqual(r['value'],sources[r['result_id']]['value'])
            self.assertEqual(r['protocol_status'],sources[r['result_id']]['protocol_status'])

    def test_comparison_protocol_columns_preserve_reported_values(self):
        paper=Path(__file__).resolve().parents[1]
        config=json.loads((paper/'data/table_spec.json').read_text())
        records=m.read_literature(paper/'data/literature_results.csv')
        row=next(r for r in config['main_rows'] if r['row_id']=='rskt_seg_dlrsd_large')
        cells=m.comparison_cells(row,records,{},config,[])
        self.assertEqual(cells[1:3],['54.32','--'])
        self.assertIn(r'$^{8}$',cells[6])
        row=dict(row,reported_protocols={'iSAID':'16-BG','UAVid':7})
        cells=m.comparison_cells(row,records,{},config,[])
        self.assertEqual(cells[1:3],['--','54.32'])
        self.assertIn(r'$^{7}$',cells[6])
        row['reported_protocols']={}
        cells=m.comparison_cells(row,records,{},config,[])
        self.assertEqual(cells[1:3],['--','--'])
        note=m.isaid_reference_note([row],records,{},config)
        self.assertIn('RSKT-Seg: 54.32',note)
        self.assertEqual((' & '.join(cells)+note).count('54.32'),1)
        self.assertIn(r'$^{u}$',cells[6])
        row['reported_protocols']={'iSAID':'guessed'}
        with self.assertRaises(m.DataError):m.comparison_cells(row,records,{},config,[])
        full=next(r for r in config['main_rows'] if r['row_id']=='full')
        missing=[]
        self.assertEqual(m.comparison_cells(full,records,{},config,missing)[1:3],['--','--'])
        self.assertEqual({r['dataset'] for r in missing if r['dataset'].startswith('iSAID')},
                         {'iSAID-15-IgnBG','iSAID-16-BG'})

    def test_every_main_row_has_two_independent_isaid_cells(self):
        paper=Path(__file__).resolve().parents[1]
        config=json.loads((paper/'data/table_spec.json').read_text())
        records=m.read_literature(paper/'data/literature_results.csv')
        for row in config['main_rows']:
            cells=m.comparison_cells(row,records,{},config,[])
            self.assertEqual(len(cells),9,row['row_id'])
            self.assertNotIn(r'\multicolumn',' & '.join(cells),row['row_id'])
        note=m.isaid_reference_note(config['main_rows'],records,{},config)
        for name,value in [('Pi-Seg','53.50'),('UGDMoE','54.38'),('SkySense-O','43.92'),
                           ('ConInfer','20.08'),('GLoP-SAM3 (ROSE report)','28.00')]:
            self.assertEqual(note.count(f'{name}: {value}'),1)
        main=(paper/'tables/main_rskt8.tex').read_text()
        rows=[line for line in main.splitlines() if ' & ' in line and not line.startswith('Method &')]
        self.assertEqual(len(rows),25)
        self.assertTrue(all(line.count(' & ')==10 for line in rows))

    def test_isaid_dual_protocol_reuses_weights_but_not_protocol_id(self):
        paper=Path(__file__).resolve().parents[1]
        config=json.loads((paper/'data/table_spec.json').read_text())
        full=next(r for r in config['main_rows'] if r['row_id']=='full')
        selector=dict(eval_ids=['one'],expected_seeds=[42],checkpoint_sha256_by_seed={'42':'a'*64},
                      eval_protocol_id='ignore-bg',eval_split='val')
        full['native_selection']={'iSAID-15-IgnBG':selector,
                                  'iSAID-16-BG':dict(selector,eval_ids=['two'],eval_protocol_id='score-bg')}
        m.validate_shared_selections(config)
        full['native_selection']['iSAID-16-BG']['checkpoint_sha256_by_seed']={'42':'b'*64}
        with self.assertRaises(m.DataError):m.validate_shared_selections(config)
        full['native_selection']['iSAID-16-BG']=dict(selector,eval_ids=['two'])
        with self.assertRaises(m.DataError):m.validate_shared_selections(config)

    def test_experiment_plan_and_chapter_cover_the_same_design(self):
        paper=Path(__file__).resolve().parents[1]
        config=json.loads((paper/'data/table_spec.json').read_text())
        with (paper/'data/experiment_plan.csv').open() as stream:
            plan=list(csv.DictReader(stream))
        training={r['variant_id'] for r in plan if r['retrain']=='true'}
        displayed={r['variant_id'] for r in config['ablation_rows']+config['freeze_rows']}
        self.assertEqual(training,displayed)
        self.assertEqual(len(training),10)
        self.assertEqual({r['variant_id'] for r in plan if r['kind']=='inference'},
                         {'semantic_only','semantic_presence'})
        self.assertEqual(len(config['freeze_rows']),4)
        self.assertEqual(len(config['ablation_rows']),7)
        chapter=(paper/'sections/experiments.tex').read_text()
        for name in ('main_rskt8','loveda_per_class','ablation_core','ablation_freeze','efficiency'):
            self.assertIn('\\input{tables/'+name+'}',chapter)
        self.assertNotIn('tab:dataset_protocol',chapter)
        self.assertNotIn('2e-6',chapter)
        self.assertNotIn('四个完整块',chapter)

    def test_source_supervision_and_citations_are_not_silently_relabelled(self):
        import re
        paper=Path(__file__).resolve().parents[1]
        config=json.loads((paper/'data/table_spec.json').read_text())
        records=m.read_literature(paper/'data/literature_results.csv')
        bib=(paper/'references.bib').read_text()+(paper/'references_extra.bib').read_text()
        keys=set(re.findall(r'@\w+\{([^,]+)',bib))
        self.assertEqual(len(config['main_rows']),25)
        self.assertNotIn('additional_rows',config)
        self.assertFalse({'gfmr','cb','prs'} & {r['method_id'] for r in config['main_rows']})
        for row in config['main_rows']:
            if row.get('citekey'):self.assertIn(row['citekey'],keys)
            for rid in row.get('literature_record_ids',{}).values():
                r=records[rid]
                self.assertIn(r['method_citekey'],keys)
                self.assertIn(r['source_citekey'],keys)
                if row['method_id']=='glop':
                    self.assertEqual(r['report_type'],'requoted_by_source')
                    self.assertEqual(r['source_citekey'],'songROSESAM3ResponseOrientedPrompt2026')
                if row['method_id']=='active':self.assertEqual(r['report_type'],'reproduced_by_source')
                if row['resource']=='R':self.assertEqual(r['training_data'],'DLRSD')

    def test_abstract_evidence_cannot_be_presented_as_an_inspected_pdf(self):
        paper=Path(__file__).resolve().parents[1]
        with (paper/'data/literature_results.csv').open() as stream:
            records=list(csv.DictReader(stream))
        r=next(r for r in records if r.get('evidence_type')=='publisher_abstract')
        def read(row):
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'source.csv'
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
                return m.read_literature(p)
        self.assertIn(r['result_id'],read(r))
        with self.assertRaises(m.DataError):read(dict(r,pdf_filename='unread.pdf'))
        with self.assertRaises(m.DataError):read(dict(r,protocol_status='reference_only'))
        with self.assertRaises(m.DataError):read(dict(r,source_url='https://example.org/abstract'))

    def test_per_class_scores_use_the_same_multiseed_selection(self):
        names=['background','building','road','water','barren','forest','agricultural']
        records={}
        for seed,score in ((1,0.2),(2,0.4)):
            records[str(seed)]=dict(source='native',record_type='single_run',role='evaluation',status='completed',
                is_full='true',result_path='/tmp/synthetic/result.json',seed=str(seed),evaluated_samples='2',
                expected_samples='2',checkpoint_sha256=str(seed)*64,method_id='reca_sam3',variant_id='full',
                training_dataset='dlrsd',eval_dataset='loveda',eval_protocol_id='p',eval_split='val',
                miou_fraction=str(score),class_names=json.dumps(names),
                per_class=json.dumps([dict(class_id=i,class_name=n,iou=score,union=1) for i,n in enumerate(names)]))
        selector=dict(eval_ids=['1','2'],expected_seeds=[1,2],checkpoint_sha256_by_seed={'1':'1'*64,'2':'2'*64},
                      eval_protocol_id='p',eval_split='val')
        config=dict(training_dataset='dlrsd',dataset_aliases={'loveda':'LoveDA'},native_decimals=2,std_ddof=1)
        result=m.native_value(selector,records,dict(method_id='reca_sam3',variant_id='full'),'LoveDA',config,'iou.road')
        self.assertEqual(result,(r'30.00 $\pm$ 14.14',30.0))

    def test_measurement_from_standard_json(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'result.json';h='c'*64
            path.write_text(json.dumps({'eval_id':'measure','checkpoint':{'sha256':h},'efficiency':{'latency_ms':0}}))
            r={'source':'native','status':'completed','role':'evaluation','is_full':'true','result_path':str(path),'checkpoint_sha256':h}
            selected={'eval_id':'measure','checkpoint_sha256':h,'json_path':'efficiency.latency_ms','display_scale':1}
            self.assertEqual(m.measurement(selected,{'measure':r},'P00','time_ms',[]),'0.00')
            selected['checkpoint_sha256']='d'*64
            with self.assertRaises(m.DataError):m.measurement(selected,{'measure':r},'P00','time_ms',[])
            missing=[]
            self.assertEqual(m.measurement(None,{},'P00','time_ms',missing),'--')
            self.assertEqual(missing[0]['reason'],'planned_measurement')

    def test_explicit_selection_and_loveda_policy(self):
        p=Path(__file__).resolve().parents[1]/'data/literature_results.csv'
        with p.open() as stream:rows=list(csv.DictReader(stream))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'input.csv'
            def write(data):
                with out.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(data)
            write([rows[0],rows[0]])
            with self.assertRaises(m.DataError):m.read_literature(out)
            love=next(dict(r) for r in rows if r['result_id']=='ov_original_loveda')
            love['class_count']='6';write([love])
            with self.assertRaises(m.DataError):m.read_literature(out)
            love['class_count']='7';love['value']='NaN';write([love])
            with self.assertRaises(m.DataError):m.read_literature(out)
        row=dict(row_id='r',method_id='x',variant_id='y',literature_record_ids={'LoveDA':['first','second']})
        with self.assertRaises(m.DataError):m.resolve(row,'LoveDA',{}, {}, {}, [])

if __name__=='__main__':unittest.main()


