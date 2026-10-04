import unittest
import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from periodic_graph_v3 import periodic_edges, PeriodicGatedConv, mean_pool


class GraphTests(unittest.TestCase):
    def test_single_atom_periodic_neighbors(self):
        st=Structure(Lattice.cubic(3),['Ca'],[[0,0,0]])
        src,dst,images,distance,_=periodic_edges(st,3.01)
        self.assertEqual(len(src),6)
        self.assertTrue(np.allclose(distance,3))
        self.assertTrue(np.all(np.any(images!=0,axis=1)))

    def test_triclinic_matches_neighbor_oracle(self):
        st=Structure(Lattice.from_parameters(4,5,6,70,80,60),['Ca','O'],[[0,0,0],[.9,.8,.7]])
        src,dst,images,dist,_=periodic_edges(st,4)
        expected=sorted((i,n.index,round(n.nn_distance,7)) for i,ns in enumerate(st.get_all_neighbors(4)) for n in ns)
        actual=sorted((int(i),int(j),round(float(d),7)) for i,j,d in zip(src,dst,dist))
        self.assertEqual(actual,expected)

    def test_neighbor_identity_affects_message(self):
        torch.manual_seed(42)
        conv=PeriodicGatedConv(4,2)
        nodes=torch.zeros(2,4)
        src=torch.tensor([0]); dst=torch.tensor([1]); edges=torch.ones(1,2)
        before=conv(nodes,edges,src,dst)[0]
        nodes[1]=1
        self.assertFalse(torch.allclose(before,conv(nodes,edges,src,dst)[0]))

    def test_permutation_equivariance_and_pooling(self):
        torch.manual_seed(42)
        conv=PeriodicGatedConv(4,2)
        nodes=torch.randn(3,4); edges=torch.randn(4,2)
        src=torch.tensor([0,1,1,2]); dst=torch.tensor([1,0,2,1])
        perm=torch.tensor([2,0,1]); inverse=torch.argsort(perm)
        expected=conv(nodes,edges,src,dst)[perm]
        actual=conv(nodes[perm],edges,inverse[src],inverse[dst])
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6))
        self.assertTrue(torch.allclose(mean_pool(nodes,[3]),mean_pool(nodes.repeat(2,1),[6])))

if __name__=='__main__':
    unittest.main()
