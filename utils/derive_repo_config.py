import json
import logging
import yaml
from pathlib import Path
from envs import KnobClass, SettingType, is_knob_enum, resolve_enum_value
from envs.spaces.utils import fetch_server_knobs
import logging


def _derive_booster_config(env, repo):
    with open(repo) as f:
        bconfig = json.load(f)

    tqmap = {b["qidx"]: Path(b["qfile"]).parts[-1] for b in bconfig["targets"]}
    bcardinal = sorted(bconfig["cardinals"], key=lambda x: x["total_rc"])[0]

    # Adjust the fillfactor to Proto-X numbers.
    indexes = []
    for idx in bcardinal["indexes"]:
        if "fillfactor" in idx:
            orig_idx = idx
            assert " WITH " in idx
            wsplits = idx.split(" WITH (")
            tsplits = wsplits[1].split(")")

            wclause = tsplits[0]
            wclause = wclause.replace(" ", "")
            ff = wclause.split("fillfactor=")[1].split(",")[0]
            assert f"fillfactor={ff}" in wclause
            nff = "90" if int(ff) <= 90 else "100"
            wclause = wclause.replace(f"fillfactor={ff}", f"fillfactor={nff}")
            idx = wsplits[0] + " WITH (" + wclause + ")" + tsplits[1]
            logging.debug(f"Transmuting {orig_idx} -> {idx}")
        indexes.append(idx)

    # Generate knobs...
    knobs = fetch_server_knobs(
        env.connection,
        env.env_spec.tables,
        env.action_space.get_knob_space().knobs,
        workload=env.workload,
    )

    for k, v in bcardinal["sysknobs"].items():
        if k in knobs:
            knobs[k] = v

    for qconfig in bcardinal["qconfigs"]:
        tqcands = [
            stem
            for stem, sfile in env.workload.sql_mapping.items()
            if Path(sfile).parts[-1] == tqmap[qconfig["qidx"]]
        ]
        assert len(tqcands) == 1
        tqc = tqcands[0]

        qknobs = qconfig["qknobs{}".format(qconfig["qidx"])]
        for qk, qv in qknobs.items():
            if "Access Method" in qk:
                if "alias of" in qk:
                    aname = qk.split(" for ")[-1].split("(")[0].strip()
                else:
                    aname = qk.split(" for ")[-1].strip()
                key = f"{tqc}_{aname}_scanmethod"
                assert env.action_space.get_knob_space().knobs[key].knob_type == SettingType.SCANMETHOD_ENUM_CATEGORICAL
                value = {
                    "Index Scan": 2,
                    "Bitmap Scan": 1,
                    "Seq Scan": 0,
                }[qv]

            elif "Force Parallel" in qk:
                key = f"{tqc}_{tqc}_parallel_rel"
                if key not in env.action_space.get_knob_space().knobs:
                    logging.warn(f"Unable to definitively warp {key} due to missing.")
                    continue

                kvalues = env.action_space.get_knob_space().knobs[key].values
                if qv == "None" or qv == "Default":
                    value = 0
                else:
                    value = None
                    for i, v in enumerate(kvalues):
                        if v == qv:
                            value = i + 1

                    if value is None:
                        # Try to alias map.
                        if qv in env.workload.query_aliases[tqc] and len(env.workload.query_aliases[tqc][qv]) == 1:
                            tv = env.workload.query_aliases[tqc][qv][0]
                            for i, v in enumerate(kvalues):
                                if v == tv:
                                    value = i + 1

                    if value is None:
                        logging.warn(f"Unable to definitively warp {key} {kvalues} {qv} {env.workload.query_aliases[tqc]}")
                        if qv in env.workload.query_aliases[tqc] and len(env.workload.query_aliases[tqc][qv]) > 1:
                            tv = env.workload.query_aliases[tqc][qv][0]
                            for i, v in enumerate(kvalues):
                                if v == tv:
                                    value = i + 1

                        if value is None:
                            value = 0

            elif "Materialize or Inline" in qk:
                cte = qk.split(" CTE ")[-1]
                key = f"{tqc}__ctemat_{cte}"
                value = {
                    "Materialize": 2,
                    "Inline": 1,
                    "Default": 0,
                }[qv]

            elif qk == "dkid":
                continue

            else:
                key = f"{tqc}_{qk}"
                value = qv

            assert key in knobs, print(key)
            logging.info(f"Loading {key} -> {value}")
            knobs[key] = value

    return knobs, indexes


def derive_repo_config(env, repo, templatize, benchmark, noop_index=False, use_booster=False):
    assert Path(repo).exists()
    if use_booster:
        assert not templatize
        return _derive_booster_config(env, repo)

    with open(Path(repo).parent.parent / f"{benchmark}.yaml") as f:
        data = yaml.safe_load(f)["mythril"]["query_spec"]
        qorder = data.get("execute_query_order", data["query_order"])
    with open(qorder) as f:
        orig_qmap = {k.split(",")[0]: k.split(",")[1].strip() for k in f.readlines()}

    index_sqls = []
    knobs = {}
    insert_knobs = False
    with open(f"{repo}/act_sql.txt", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                insert_knobs = True
            elif not insert_knobs:
                index_sqls.append(line)
            else:
                k, v = line.split(" = ")
                if k.startswith("Q"):
                    qid = k.split("_")[0]
                    assert qid in orig_qmap, print(qid, orig_qmap)
                    if templatize:
                        def _partify(sqlfile):
                            sqlfile = str(sqlfile)
                            if sqlfile.endswith(".sql"):
                                sqlfile = sqlfile[:-4]
                            spj = "_spj" in sqlfile
                            qpiece = sqlfile.split("s")[0]
                            return (qpiece, spj)

                        match_stems = [
                            k
                            for k, v in env.workload.sql_mapping.items()
                            if _partify(str(v).split("/")[-1].strip()) == _partify(orig_qmap[qid].strip())
                        ]
                    else:
                        # Get matching stems that match SQL file exactly.
                        match_stems = [
                            k
                            for k, v in env.workload.sql_mapping.items()
                            if str(v).split("/")[-1].strip() == orig_qmap[qid].strip()
                        ]
                        #logging.info(f"Mapping {qid}-{orig_qmap[qid]} to {match_stems}-{[env.workload.sql_mapping[k] for k in match_stems]}")

                    # Swap the QID into the corresponding matching stem.
                    # If there's no matching stem, ignore it.
                    for match_stem in match_stems:
                        kk = k.replace(f"{qid}_", f"{match_stem}_")
                        if "_scanmethod" in k or "_ctemat" in k or "_parallel_rel" in k:
                            knobs[kk] = int(v)
                        else:
                            knobs[kk] = float(v)
                else:
                    if k in env.action_space.get_knob_space().knobs:
                        knob = env.action_space.get_knob_space().knobs[k]
                        if knob.knob_type == SettingType.FLOAT:
                            knobs[k] = float(v)
                        else:
                            knobs[k] = int(v)
                    else:
                        knobs[k] = float(v)

    assert len(index_sqls) > 0
    assert len(knobs) > 0
    with open(f"{repo}/prior_state.txt", "r") as f:
        prior_states = eval(f.read())
        all_sc = [s.strip() for s in prior_states[1]]
        if not noop_index:
            all_sc.extend(index_sqls)
        all_sc = [a for a in all_sc if not "USING btree ()" in a]
        index_sqls = all_sc

    isc = set()
    nis = []
    for isql in index_sqls:
        t = isql.lower().split(" on ")[-1].strip()
        if t in isc:
            continue

        isc.add(t)
        nis.append(isql)

    return knobs, nis
