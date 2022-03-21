import os

from const import LOGDIR, PATIENTS


for p in PATIENTS:
	os.system('python3 tb_extract_seg.py --patient ' + p + ' 2>&1 | tee ' + os.path.join(LOGDIR, 'log_seg_' + p + '.txt'))