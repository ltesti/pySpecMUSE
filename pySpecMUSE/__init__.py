# SPDX-License-Identifier: LGPL-3.0-or-later
#
# This file is part of pySpecMUSE.
# Copyright (C) 2023-2026 the pySpecMUSE authors (see README)
#
# pySpecMUSE is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pySpecMUSE is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pySpecMUSE (files COPYING.LESSER and COPYING). If not, see
# <https://www.gnu.org/licenses/>.

from ._version import __version__

import matplotlib

# To avoid Runtime Error
# RuntimeError: Python is not installed as a framework. The Mac OS X backend
# will not be able to function correctly if Python is not installed as a framework.
# See the Python documentation for more information on installing Python as a
# framework on Mac OS X. Please either reinstall Python as a framework, or try
# one of the other backends.
# If you are using (Ana)Conda please install python.app and replace the use of
# 'python' with 'pythonw'.
# See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.
# https://matplotlib.org/faq/osx_framework.html

if matplotlib.get_backend().lower() == 'macosx':
    matplotlib.use('TkAgg')

from .myphotutils import get_centroids, get_stars_for_apc, runphot_ima_aps, do_apphot, apc_spec_single_wl
from .apc_plots import plot_curve_of_growth_iv, plot_apc
from .aperturesplots import plotstars
from .utils import running_median_spec
from .CubeMUSE import CubeMUSE
from .StarMUSE import StarMUSE
from .star_plots import plot_star
from .lines import line_integral, lineplot, known_lines
