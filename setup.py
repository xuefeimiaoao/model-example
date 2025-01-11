#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2025 [xuefeimiaoao](https://github.com/xuefeimiaoao). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for palette."""

from __future__ import absolute_import
import io
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with io.open(os.path.join("nlp", "__init__.py"), "rt", encoding='utf-8') as f:
    SDK_VERSION = re.search(r"SDK_VERSION = b'(.*?)'", f.read()).group(1)

with io.open(os.path.join("nlp/environments", "requirements.txt"), "rt", encoding='utf-8') as f:
    REQUIRED_PACKAGES = f.read()

setup(
    name='nlp_examples',
    version=SDK_VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=['nlp'],
    platforms="any",
    python_requires='>=3.7',

    description='Examples for model training',
    long_description='',
    author='',
    author_email='xuefeimiaoao@gmail.com',
    url='https://github.com/xuefeimiaoao/model-example',
    license='Apache License 2.0'
)
