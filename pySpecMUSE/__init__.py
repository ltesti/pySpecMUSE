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

from .myphotutils import get_centroids, get_stars_for_apc
from .aperturesplots import plotstars
from .CubeMUSE import CubeMUSE
