import os
import sys
cur_folder= os.path.dirname(os.path.abspath(__file__))
if cur_folder not in sys.path:
    sys.path.append(cur_folder)
