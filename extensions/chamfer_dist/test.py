# -*- coding: utf-8 -*-
# @Author: Ajay Narasimha Mopidevi
# @Date:   2025-11-01 20:54:24
# @Email: ajaynmopidevi@gmail.com
# Adapted from Haozhe Xie Version 2.0.0

# Previous Version
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in chamfer.cu

import os
import sys
import torch
import unittest


from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.chamfer_dist import ChamferFunction


class ChamferDistanceTestCase(unittest.TestCase):
    def test_chamfer_dist(self):
        x = torch.rand(4, 64, 3).double()
        y = torch.rand(4, 128, 3).double()
        x.requires_grad = True
        y.requires_grad = True
        print(gradcheck(ChamferFunction.apply, [x.cuda(), y.cuda()]))



if __name__ == '__main__':
    # unittest.main()
    import pdb
    x = torch.rand(32,128,3)
    y = torch.rand(32,128,3)
    pdb.set_trace()
