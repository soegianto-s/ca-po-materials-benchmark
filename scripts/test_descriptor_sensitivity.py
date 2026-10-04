"""Guard the changed cohort and evaluation logic against leakage and misalignment."""
import json,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from descriptor_sensitivity_experiment import EXCLUDE,D,ROOT
from summarize_descriptor_sensitivity import pool,measure
class Tests(unittest.TestCase):
 def test_filtered_partitions_preserve_membership_and_composition_separation(self):
  df=pd.read_csv(ROOT/'processed/15_ml_magpie_features.csv').set_index('material_id')
  for splitfile in D.glob('partition*/split.json'):
   new=json.loads(splitfile.read_text());old=json.loads((D.parent/'physics_hybrid_v1'/splitfile.parent.name/'split.json').read_text())
   for part in ['fit','validation','test']:
    self.assertEqual(new[part],[x for x in old[part] if x not in EXCLUDE])
    self.assertEqual(set(new[part+'_groups']),{Composition(df.loc[x,'formula']).reduced_formula for x in new[part]})
   self.assertEqual(len(set(new['fit']+new['validation']+new['test'])),508)
   for a,b in [('fit','test'),('fit','validation'),('validation','test')]:self.assertFalse(set(new[a+'_groups'])&set(new[b+'_groups']))
 def test_exclusion_preserves_threshold_and_prediction_alignment(self):
  ids=['a','mp-1214043','b'];row=dict(ids=ids,truth=[[0,1,1,0],[1,1,99,0],[1,1,2,0]],probability=[.2,.99,.8],calibrated_probability=[.3,.9,.7],band_gap=[1,999,2],formation_energy=[0,999,0],analytic_density=[1,1,1],threshold=.4,calibrated_threshold=.6)
  one=pool([row],2);self.assertEqual(one['ids'],['a','b']);self.assertEqual(one['threshold'],[.4,.4]);self.assertEqual(one['calibrated_threshold'],[.6,.6]);self.assertEqual(measure(one)['band_gap_mae'],0)
  two=pool([one],2);self.assertEqual(two,one)
 def test_retained_fraction_columns_are_complete(self):
  df=pd.read_csv(ROOT/'processed/15_ml_magpie_features.csv');x=df.loc[~df.material_id.isin(EXCLUDE),[c for c in df if c.startswith('frac_')]]
  np.testing.assert_allclose(x.sum(axis=1),1,rtol=0,atol=1e-12)
if __name__=='__main__':unittest.main()
