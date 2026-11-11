"""Boundary checks for the paper renderer; all synthetic records stay in memory/temp dirs."""
import csv
import importlib.util
import json
import tempfile
import unittest
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
        self.assertEqual(m.average({'LoveDA':(display,value,matched)},['LoveDA'],True),'--')
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

    def test_missing_or_unlocked_average(self):
        self.assertEqual(m.average({'a':('0',0,True),'b':('50',50,True)},['a','b'],True),'25.00')
        self.assertEqual(m.average({'a':('--',None,True)},['a'],True),'--')
        self.assertEqual(m.average({'a':('10',10,False)},['a'],True),'--')
        self.assertEqual(m.average({'a':('10',10,True)},['a'],False),'--')

    def test_native_conversion_and_duplicate_seed(self):
        h='a'*64
        names=['background','building','road','water','barren','forest','agricultural']
        r=dict(eval_id='e1',source='native',record_type='single_run',role='evaluation',status='completed',is_full='true',result_path='/tmp/synthetic/result.json',seed='9',evaluated_samples='2',expected_samples='2',checkpoint_sha256=h,method_id='reca_sam3',variant_id='full',training_dataset='isaid',eval_dataset='loveda',eval_protocol_id='p',eval_split='test',miou_fraction='0',class_names=json.dumps(names),per_class=json.dumps([dict(class_id=i,class_name=n,iou=0,union=1) for i,n in enumerate(names)]))
        selector=dict(eval_ids=['e1'],expected_seeds=[9],checkpoint_sha256_by_seed={'9':h},eval_protocol_id='p',eval_split='test')
        config=dict(training_dataset='isaid',dataset_aliases={'loveda':'LoveDA'},native_decimals=2,std_ddof=1)
        row=dict(method_id='reca_sam3',variant_id='full')
        def resolve():return m.native_value(selector,records,row,'LoveDA',config)
        records={'e1':r}
        self.assertEqual(resolve(),('0.00',0))
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

