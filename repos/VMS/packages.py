import os
import sys


def include():
    sys.path.append(os.path.abspath(os.path.join('..', 'common')))
    sys.path.append(os.path.abspath(os.path.join('ai_core/object_detection','plate_character')))
#    sys.path.append(os.path.abspath(os.path.join('..', 'common')))
