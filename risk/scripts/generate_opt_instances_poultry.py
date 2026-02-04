DESC = '''
Generate model instances for dairy risk.

AA
'''

from itertools import product
import numpy as np
import os
import pandas as pd
from pdb import set_trace
import stat

with open('run', 'w') as f:
    for i in range(100):
        f.write(f'''sbatch -o opt_{i} \
--export=ALL,command=\"python ../scripts/optimize_poultry.py -s {i}\" \
../scripts/run_proc.sbatch; qreg;\n''')

os.chmod('run', os.stat('run').st_mode | stat.S_IXUSR)

