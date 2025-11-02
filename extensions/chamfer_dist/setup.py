# -*- coding: utf-8 -*-
# @Author: Ajay Narasimha Mopidevi
# @Date:   2025-11-01 20:54:24
# @Email: ajaynmopidevi@gmail.com
# Adapted from Haozhe Xie Version 2.0.0

# Previous Version
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='chamfer',
      version='3.0.0',
      ext_modules=[
          CUDAExtension('chamfer', [
              'chamfer_cuda.cpp',
              'chamfer.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
