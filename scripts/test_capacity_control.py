import unittest
import torch
from capacity_control_experiment import CapacityModel,Original
from test_training_benchmark import graph
from train_periodic_benchmark import batch
from pymatgen.core import Structure,Lattice
class Tests(unittest.TestCase):
 def test_capacity_invariance_and_no_composition_input(self):
  torch.set_num_threads(1);torch.manual_seed(42)
  m=CapacityModel().eval();old=Original(True)
  self.assertEqual(sum(p.numel() for p in m.parameters()),sum(p.numel() for p in old.parameters()))
  st=Structure(Lattice.cubic(4),['Ca','O'],[[0,0,0],[.5,.5,.5]])
  gs=[graph(st),graph(st*2)];args=batch(gs,[0,1])
  out=m(*args,torch.randn(2,90));other=m(*args,torch.randn(2,90))
  self.assertTrue(torch.equal(out,other));self.assertTrue(torch.allclose(out[0],out[1],atol=1e-5))
  out[:,[0,2,3]].sum().backward()
  self.assertTrue(all(p.grad is not None for p in m.parameters()))
  self.assertGreater(m.composition[0].weight.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
