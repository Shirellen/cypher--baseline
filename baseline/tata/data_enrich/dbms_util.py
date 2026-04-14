import psycopg2

stats_config = {
    'database' : 'stats',
    'user' : 'zy',
    'password' : 'zy',
    'port' : 5434,
}

_ALL_OPTIONS = [
    "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
    "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"
]
keys = [['hashjoin', 'indexonlyscan'],
 ['hashjoin', 'indexonlyscan', 'indexscan'],
 ['hashjoin', 'indexonlyscan', 'indexscan', 'mergejoin'],
 ['hashjoin', 'indexonlyscan', 'indexscan', 'mergejoin', 'nestloop'],
 ['hashjoin', 'indexonlyscan', 'indexscan', 'mergejoin', 'seqscan'],
 ['hashjoin', 'indexonlyscan', 'indexscan', 'nestloop'],
 ['hashjoin', 'indexonlyscan', 'indexscan', 'nestloop', 'seqscan'],
 ['hashjoin', 'indexonlyscan', 'indexscan', 'seqscan'],
 ['hashjoin', 'indexonlyscan', 'mergejoin'],
 ['hashjoin', 'indexonlyscan', 'mergejoin', 'nestloop'],
 ['hashjoin', 'indexonlyscan', 'mergejoin', 'nestloop', 'seqscan'],
 ['hashjoin', 'indexonlyscan', 'mergejoin', 'seqscan'],
 ['hashjoin', 'indexonlyscan', 'nestloop'],
 ['hashjoin', 'indexonlyscan', 'nestloop', 'seqscan'],
 ['hashjoin', 'indexonlyscan', 'seqscan'],
 ['hashjoin', 'indexscan'],
 ['hashjoin', 'indexscan', 'mergejoin'],
 ['hashjoin', 'indexscan', 'mergejoin', 'nestloop'],
 ['hashjoin', 'indexscan', 'mergejoin', 'nestloop', 'seqscan'],
 ['hashjoin', 'indexscan', 'mergejoin', 'seqscan'],
 ['hashjoin', 'indexscan', 'nestloop'],
 ['hashjoin', 'indexscan', 'nestloop', 'seqscan'],
 ['hashjoin', 'indexscan', 'seqscan'],
 ['hashjoin', 'mergejoin', 'nestloop', 'seqscan'],
 ['hashjoin', 'mergejoin', 'seqscan'],
 ['hashjoin', 'nestloop', 'seqscan'],
 ['hashjoin', 'seqscan'],
 ['indexonlyscan', 'indexscan', 'mergejoin'],
 ['indexonlyscan', 'indexscan', 'mergejoin', 'nestloop'],
 ['indexonlyscan', 'indexscan', 'mergejoin', 'nestloop', 'seqscan'],
 ['indexonlyscan', 'indexscan', 'mergejoin', 'seqscan'],
 ['indexonlyscan', 'indexscan', 'nestloop'],
 ['indexonlyscan', 'indexscan', 'nestloop', 'seqscan'],
 ['indexonlyscan', 'mergejoin'],
 ['indexonlyscan', 'mergejoin', 'nestloop'],
 ['indexonlyscan', 'mergejoin', 'nestloop', 'seqscan'],
 ['indexonlyscan', 'mergejoin', 'seqscan'],
 ['indexonlyscan', 'nestloop'],
 ['indexonlyscan', 'nestloop', 'seqscan'],
 ['indexscan', 'mergejoin'],
 ['indexscan', 'mergejoin', 'nestloop'],
 ['indexscan', 'mergejoin', 'nestloop', 'seqscan'],
 ['indexscan', 'mergejoin', 'seqscan'],
 ['indexscan', 'nestloop'],
 ['indexscan', 'nestloop', 'seqscan'],
 ['mergejoin', 'nestloop', 'seqscan'],
 ['mergejoin', 'seqscan'],
 ['nestloop', 'seqscan']]


def arm_idx_to_hints(arm_idx):
    hints = []
    for option in _ALL_OPTIONS:
        hints.append(f"SET {option} TO off")

    if arm_idx == 0:
        for option in _ALL_OPTIONS:
            hints.append(f"SET {option} TO on")
    elif arm_idx < 49:
        arm_keys = keys[arm_idx-1]
        for key in arm_keys:
            hints.append(f"SET enable_{key} TO on")
    else:
        raise Exception("RegBlocker only supports the 48+1 arms")
    return hints


def get_bao_plans(db_config, sqls):
    conm = psycopg2.connect(database=db_config['database'],user=db_config['user'],
        password=db_config['password'],port=db_config['port'])
    conm.commit()
    conm.autocommit = True
    cur = conm.cursor()

    # to_add_planss = []
    plans = []
    for sql in sqls:
        # plans = []
        for arm in range(49):
            for stmt in arm_idx_to_hints(arm):
                cur.execute(stmt)
                
            cmd = 'explain (analyze false, costs true, format json) '+sql
            cur.execute(cmd)
            res_json = cur.fetchall()[0][0][0]
            plans.append(res_json)
        # to_add_planss.append(plans)

    return plans