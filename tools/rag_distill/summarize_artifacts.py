import json,sys
from collections import Counter
for p in sys.argv[1:]:
 rows=[json.loads(x) for x in open(p,encoding='utf-8') if x.strip()]
 print(p,len(rows),Counter((r.get('metadata',{}).get('language'),r.get('metadata',{}).get('task_domain'),r.get('metadata',{}).get('question_type')) for r in rows))
