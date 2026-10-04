"""Check physical constraints, isolation of loss terms, and calibration mapping."""
import unittest
import numpy as np
import torch
from physics_hybrid_experiment import HybridModel,decode,objective,scale_fit,platt_fit,calibrate,VARIANTS
from train_periodic_benchmark import GraphModel,batch
from test_training_benchmark import graph
from pymatgen.core import Lattice,Structure

class PhysicsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_bounded_gap_in_physical_units_and_zero(self):
        out=torch.tensor([[0.,0.,-10.,0.],[0.,0.,2.,0.]],requires_grad=True)
        mu=torch.tensor([3.,2.,-1.]);sd=torch.tensor([1.,3.,.5])
        pred=decode(out,mu,sd,True)
        self.assertEqual(pred[0,1].item(),0.);self.assertEqual(pred[1,1].item(),8.)
        pred[:,1].sum().backward();self.assertGreater(out.grad[1,2].item(),0.)

    def test_density_loss_removal(self):
        out=torch.randn(3,4,requires_grad=True);y=torch.tensor([[1.,3.,2.,-1.],[0.,4.,1.,-2.],[0.,2.,0.,-1.]])
        args=(out,y,torch.tensor([3.,1.,-1.]),torch.ones(3),torch.tensor(2.))
        loss=objective(*args,VARIANTS['graph_physics']);loss.backward()
        self.assertTrue(torch.equal(out.grad[:,1],torch.zeros(3)))

    def test_common_selection_ignores_density(self):
        out=torch.randn(3,4);y=torch.randn(3,4);y[:,0]=torch.tensor([0.,1.,0.])
        args=(out,y,torch.zeros(3),torch.ones(3),torch.tensor(2.))
        self.assertTrue(torch.equal(objective(*args,VARIANTS['graph_gap'],True),objective(*args,VARIANTS['graph_physics'],True)))

    def test_scaling_fit_only(self):
        x=np.array([[1.,2.],[3.,2.],[1000.,1000.]])
        z,mu,sd=scale_fit(x,np.array([0,1]));self.assertTrue(np.array_equal(mu,[2.,2.]));self.assertTrue(np.array_equal(sd,[1.,1.]))

    def test_monotone_calibration(self):
        p=np.array([.1,.2,.4,.6,.8,.9]);y=np.array([0.,0.,1.,0.,1.,1.]);cal=platt_fit(y,p)
        q=calibrate(p,cal);self.assertTrue(np.all(np.diff(q)>=0));self.assertTrue(np.all((q>=0)&(q<=1)))

    def test_baseline_architecture_equivalence(self):
        st=Structure(Lattice.cubic(4),['Ca','O'],[[0,0,0],[.5,.5,.5]])
        gs=[graph(st)];torch.manual_seed(42);old=GraphModel().eval();torch.manual_seed(42);new=HybridModel(False).eval()
        with torch.no_grad():self.assertTrue(torch.equal(old(*batch(gs,[0])),new(*batch(gs,[0]),torch.zeros(1,90))))

    def test_hybrid_uses_composition_and_preserves_supercell(self):
        st=Structure(Lattice.cubic(4),['Ca','O'],[[0,0,0],[.5,.5,.5]])
        gs=[graph(st),graph(st*2)];torch.manual_seed(42);m=HybridModel(True).eval()
        with torch.no_grad():
            a=m(*batch(gs,[0,1]),torch.ones(2,90));b=m(*batch(gs,[0,1]),torch.zeros(2,90))
        self.assertTrue(torch.allclose(a[0],a[1],atol=2e-6));self.assertFalse(torch.allclose(a,b))

if __name__=='__main__':unittest.main()
