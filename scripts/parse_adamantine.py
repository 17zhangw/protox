import ast
import numpy as np
import psycopg
import shutil
import datetime
import logging
import time
import yaml
import os
import json
import pandas as pd
import tqdm
import argparse
import gymnasium as gym
from pathlib import Path
from dateutil.parser import parse

import sys
sys.path.append("/home/wz2/mythril")

from envs.spec import Spec
from envs.pg_env import PostgresEnv
from envs.spaces.knob import CategoricalKnob


def deserialize_list(v):
    if v.startswith("L+ "):
        return [l for l in v[3:].split(";") if len(l) > 0]
    return v


def deserialize_dict(v, coerce_int=False):
    if v.startswith("D+ "):
        vv = v[3:].split(",")
        result = {
            p.split("=")[0]: p.split("=")[1]
            for p in vv
            if p != "" and  p.split("=")[1] != "None"
        }

        if coerce_int:
            result = {k1: int(float(v)) for k1, v in result.items()}
    else:
        result = v
    return result


def deserialize_config(config, coerce_int=False):
    dconfig = {}
    for k, v in config.items():
        if not isinstance(v, str):
            dconfig[k] = v
        elif v.startswith("D+ "):
            dconfig[k] = deserialize_dict(v, coerce_int=coerce_int)
        elif v.startswith("L+ "):
            dconfig[k] = deserialize_list(v)
        else:
            dconfig[k] = v
    return dconfig


def load_from_pg(config):
    def _floatify(dc):
        kkeys = list(dc.keys())
        for kk in kkeys:
            if isinstance(dc[kk], dict):
                dc[kk] = _floatify(dc[kk])
            elif isinstance(dc[kk], int):
                dc[kk] = float(dc[kk])
            elif isinstance(dc[kk], str):
                try:
                    fval = float(dc[kk])
                    dc[kk] = fval
                except:
                    pass
        return dc
    dc = deserialize_config(json.loads(config))
    dc = _floatify(dc)

    # Reattach the CREATE INDEX.
    if "indexes" in dc:
        dc["indexes"] = [f"CREATE INDEX eindex{i} ON {idx}" for i, idx in enumerate(dc["indexes"])]

    return dc


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _swap_index_name(index, counter):
    name = index.split("CREATE INDEX ")[1].split(" ON ")[0]
    return index.replace(name, f"index{counter}")


def _dedup_index(lindex, rindex):
    lkeys = lindex.split("(")[1].split(")")[0].split(",")
    rkeys = rindex.split("(")[1].split(")")[0].split(",")

    linclude = None if "INCLUDE " not in lindex else lindex.split("INCLUDE (")[1].split(")")[0].split(",")
    rinclude = None if "INCLUDE" not in rindex else rindex.split("INCLUDE (")[1].split(")")[0].split(",")

    lwith = lindex.split(" WITH ")[-1]
    rwith = rindex.split(" WITH ")[-1]

    if all([l == r for l, r in zip(lkeys, rkeys)]):
        if len(lkeys) > len(rkeys):
            if rinclude is not None and len(set(rinclude).difference(set(linclude))) > 0:
                return None
        elif len(rkeys) > len(lkeys):
            if linclude is not None and len(set(linclude).difference(set(rinclude))) > 0:
                return None
        else:
            if linclude != rinclude:
                return None
    else:
        return None

    if "btree" not in lindex or "btree" not in rindex:
        return None

    lff = int(lwith.split("fillfactor=")[-1])
    rff = int(rwith.split("fillfactor=")[-1])
    if lff <= rff:
        return rindex
    return lindex


def evaluate(args):
    with open(args.input) as f:
        input_data = json.load(f)

    bcf = f"configs/benchmark/{args.benchmark}.yaml"
    if args.specialization:
        with open(bcf, "r") as f:
            data = yaml.safe_load(f)
            data["mythril"]["query_spec"]["execute_query_directory"] = str(args.specialization)
            data["mythril"]["query_spec"]["execute_query_order"] = f"{args.specialization}/d_order.txt"
        with open("/tmp/benchmark.yaml", "w") as f:
            yaml.dump(data, f)
        bcf = "/tmp/benchmark.yaml"

    spec = Spec(
        agent_type=None,
        seed=0,
        horizon=5,
        config_path="configs/config.yaml",
        benchmark_config_path=bcf,
        workload_timeout=0)

    iqmap = {str(v.parts[-1]): k for k, v in spec.workload.sql_mapping.items()}
    itarget_map = {}
    targets = input_data["targets"]
    for target in targets:
        qfile = target["qfile"]
        qfile = qfile.split("/")[-1]
        assert qfile in iqmap, print(qfile, iqmap.keys())
        itarget_map[target["qidx"]] = iqmap[qfile]

    env = PostgresEnv(
        spec,
        horizon=5,
        timeout=None,
        reward_utility=None,
        logger=None,
        replay=True)

    def run_sample(ql_knobs):
        samples = []
        qid_runs = []
        for i in range(args.samples):
            qid_runtimes = {}
            runtime =  spec.workload._execute_workload(
                connection=env.connection,
                workload_timeout=args.workload_timeout,
                ql_knobs=ql_knobs,
                env_spec=spec,
                blocklist=[l for l in args.blocklist.split(",") if len(l) > 0],
                metrics=False,
                dbgprint=True,
                qid_runtimes=qid_runtimes)
            samples.append(runtime)
            for qid, val in qid_runtimes.items():
                qid_runs.append({
                    "sample": i,
                    "qid": qid,
                    "runtime": val,
                })
            qid_runs.append({
                "sample": i,
                "qid": "total",
                "runtime": samples[-1],
            })
            logging.info(f"Runtime: {runtime}")

            if runtime >= args.workload_timeout:
                break

        return samples, qid_runs

    if "frontier" in input_data:
        frontier = input_data["frontier"]

    else:
        cardinals = input_data["cardinals"]
        frontier = sorted(cardinals, key=lambda x: x["total_rc"])

    slots = [int(x) for x in args.indexs.split(",")] if args.indexs else []
    if len(slots) == 0:
        slots = [x for x in range(len(frontier))]

    for slot in slots:
        config = frontier[slot]
        knobs = config["sysknobs"]

        qknobs = {}
        for qconfig in config["qconfigs"]:
            qidx = qconfig["qidx"]
            qkslot = "qknobs{}".format(qidx)
            qkknobs = qconfig[qkslot]

            for qkknob, qkvalue in qkknobs.items():
                if qkknob.startswith("enable_") or qkknob in ["hash_mem_multiplier", "seq_page_cost", "random_page_cost"]:
                    name = "{}_{}".format(itarget_map[qidx], qkknob)
                    assert name in spec.action_space.get_knob_space().knobs, print(name, spec.action_space.get_knob_space().knobs.keys())
                    if qkknob.startswith("enable"):
                        assert qkvalue in ["on", "off", "True", "False", 1.0, 0.0], print(qkvalue)
                        qknobs[name] = 1. if (qkvalue in ["on", "True", 1.0]) else 0.
                    else:
                        qknobs[name] = float(qkvalue)

                elif "Access Method for " in qkknob:
                    tbl = qkknob.split(" for ")[-1].split(" (")[0]
                    name = "{}_{}_scanmethod".format(itarget_map[qidx], tbl)
                    assert name in spec.action_space.get_knob_space().knobs, print(name, spec.action_space.get_knob_space().knobs.keys())
                    if "Bitmap" in qkvalue:
                        qknobs[name] = 1
                    elif "NoSeq" in qkvalue:
                        assert False
                    elif "Seq" in qkvalue:
                        qknobs[name] = 0
                    elif "Index" in qkvalue:
                        qknobs[name] = 2
                    else:
                        print(name, qkvalue)
                        assert False

                elif "Force Parallel" in qkknob:
                    name = "{}_{}_parallel_rel".format(itarget_map[qidx], itarget_map[qidx])
                    assert name in spec.action_space.get_knob_space().knobs, print(name, spec.action_space.get_knob_space().knobs.keys())
                    if qkvalue == "None" or qkvalue == "Default":
                        qknobs[name] = 0

                    else:
                        values = spec.action_space.get_knob_space().knobs[name].values
                        sel_i = None
                        for i, v in enumerate(values):
                            if v == qkvalue:
                                sel_i = i
                                break

                        assert sel_i is not None, print(name, qkvalue)
                        qknobs[name] = sel_i + 1

                elif "Materialize or Inline" in qkknob:
                    ctename = qkknob.split(" CTE ")[-1].strip()
                    name = "{}__ctemat_{}".format(itarget_map[qidx], ctename)
                    assert name in spec.action_space.get_knob_space().knobs, print(name, spec.action_space.get_knob_space().knobs.keys())
                    assert qkvalue in ["Default", "Inline", "Materialize"]
                    if qkvalue == "Materialize":
                        qknobs[name] = 2
                    elif qkvalue == "Inline":
                        qknobs[name] = 1
                    else:
                        qknobs[name] = 0

                elif qkknob == "dkid":
                    continue

                else:
                    assert False, print(qkknob)
        # This should reliably check that we are loading the correct knobs...
        ql_knobs = spec.action_space.get_knob_space().get_query_level_knobs(qknobs)

        execute_sqls = []
        indexes = sorted(config["indexes"], key=lambda x: int(x.split(" ON")[0].split("eindex")[-1]))
        for c, index in enumerate(indexes):
            nindex = _swap_index_name(index, c)
            inject_slot = None
            #for enum, es in enumerate(execute_sqls):
            #    cmp = _dedup_index(nindex, es)
            #    if cmp == es:
            #        inject_slot = -1
            #        print("Subsuming {} -> {}".format(nindex, es));
            #        break
            #    elif cmp == nindex:
            #        print("Replacing {} -> {}".format(es, nindex));
            #        inject_slot = enum
            #        break

            if inject_slot is None:
                #print("Adding {}".format(nindex))
                execute_sqls.append(nindex)
            elif inject_slot >= 0:
                execute_sqls[inject_slot] = nindex

        env.restore_pristine_snapshot()
        env.action_space.reset(**{"connection": env.connection, "workload": spec.workload})
        spec.workload.reset()

        # Reset snapshot.
        env.action_space.reset(connection=env.connection, workload=env.workload)
        cc, _ = env.action_space.get_knob_space().generate_plan(knobs, no_check=True)
        env.shift_state(cc, execute_sqls, dump_page_cache=True)
        env.connection = psycopg.connect("host=localhost port=5432 dbname=benchbase", prepare_threshold=None, autocommit=True)

        _, qid_runs = run_sample(ql_knobs)
        assert False
        if args.output is not None:
            out_dir = args.input.parent
            if "sscratch" in str(out_dir):
                out_dir = out_dir.parent

            data = pd.DataFrame(qid_runs)
            if "elapsed" in input_data:
                data["elapsed_sec"] = input_data["elapsed"]
            data.to_csv(out_dir / args.output, index=False)

    env.close()


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(prog="Parse Adamantine")
    #parser.add_argument("--input", type=Path, required=True)
    #parser.add_argument("--indexs", type=str, default=None)
    #parser.add_argument("--benchmark", required=True)
    #parser.add_argument("--specialization", default=None)
    #parser.add_argument("--workload-timeout", type=int)
    #parser.add_argument("--samples", type=int)
    #parser.add_argument("--blocklist", default="")
    #parser.add_argument("--output", default=None)
    #args = parser.parse_args()

    parsed_paths = set()
    while True:
        o = Path("/nfs1/exact_transfer/booster_ut/evoyage3large-pllama31_2/dsb_exact_sf10_wl_0_s1_json/baseline/output.json")

        dsbpart = [p for p in o.parts if "dsb" in p][0]
        dsbpart = dsbpart.split("_sf10")[0]

        fsscratch = [f.stem for f in (o.parent / "sscratch").rglob("scratch*.json")]
        fsscratch = sorted(fsscratch, key=lambda x: int(x.split("scratch")[-1].split(".json")[0]))[-1]
        print("Input: ", o, fsscratch)
        print("DSB Part: ", dsbpart)

        args = DotDict({
            "input": o.parent / "sscratch" / f"{fsscratch}.json",
            "indexs": "0",
            "benchmark": "dsb_revise",
            "workload_timeout": 300,
            "blocklist": "query081",
            "samples": 3,
            "output": "qid_data.csv",
            "specialization": f"/home/wz2/mythril/queries/specializations/{dsbpart}",
        })
        evaluate(args)
        parsed_paths.add(str(o))

        print("Sleeping till next batch...")
        time.sleep(900)
