"""Print the question ids (comma-separated) of rollouts that submitted no answer, for the retry rule.

The paper retries a rollout that ended without an answer up to twice, at the same message limit, and uses each
question's first answered attempt. Pass the output as `-T ids=...` to a retry run.

usage: python experiments/sandbagging/unanswered.py 'logs/note/base/*.eval'
"""

import glob
import math
import re
import sys

from inspect_ai.log import read_eval_log

ids = set()
for f in sorted(glob.glob(sys.argv[1])):
    for s in read_eval_log(f).samples or []:
        v = getattr((s.scores or {}).get("abq_correct"), "value", None)
        if v is None or math.isnan(v):
            ids.add(int(re.search(r"_q(\d+)_r", s.id).group(1)))
print(",".join(str(i) for i in sorted(ids)))
