"""Focused checks for adapter equivalence and the class-weight odds correction."""
import unittest
import numpy as np,torch
from scipy.special import expit,logit
from reviewer_extension_experiment import AdaptGraph,CompositionMLP
from physics_hybrid_experiment import HybridModel
class Checks(unittest.TestCase):
 def test_original_graph_is_preserved(self):
  torch.manual_seed(42);a=HybridModel(False).eval();torch.manual_seed(42);b=AdaptGraph(32,2,.1).eval()
  self.assertEqual(list(a.state_dict()),list(b.state_dict()))
  for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
  nodes=torch.rand(4,8);edges=torch.rand(4,8);src=torch.tensor([0,1,2,3]);dst=torch.tensor([1,0,3,2]);assignment=torch.tensor([0,0,1,1]);args=(nodes,edges,src,dst,assignment,2,torch.rand(2,90))
  torch.testing.assert_close(a(*args),b(*args),rtol=0,atol=0)
 def test_composition_only(self):
  m=CompositionMLP(32,.1).eval();x=torch.rand(3,90);self.assertEqual(tuple(m(x).shape),(3,4));m(x).sum().backward();self.assertTrue(all(p.grad is not None for p in m.parameters()))
 def test_analytic_weight_inverse(self):
  p=np.array([.001,.1,.5,.99]);w=4.;weighted=w*p/(1-p+w*p);np.testing.assert_allclose(expit(logit(weighted)-np.log(w)),p,rtol=1e-12,atol=1e-12)
if __name__=='__main__':unittest.main()
