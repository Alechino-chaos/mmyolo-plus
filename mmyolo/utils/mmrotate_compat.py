# Copyright (c) OpenMMLab. All rights reserved.
"""Compatibility aliases required by MMRotate 1.0.0rc1 on Python 3.10+."""

import collections
import collections.abc


# MMRotate 1.0.0rc1 still imports Sequence from collections in a detector
# module. Python 3.10 removed that alias, so restore it before importing
# mmrotate.models and registering its rotated components.
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence
