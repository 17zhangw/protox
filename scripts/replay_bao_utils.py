import json
import tqdm
import yaml
import pandas as pd
import logging
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def run_autosteer(env, num_samples, workload_timeout, autosteer_file, benchmark_config_path):
    with open(benchmark_config_path, "r") as f:
        qo = yaml.safe_load(f)["mythril"]["query_spec"]["query_order"]
    with open(qo, "r") as f:
        sql_mapping = {}
        for l in f:
            c = l.strip().split(",")
            sql_mapping[c[1]] = c[0]

    env.action_space.get_knob_space().reset(connection=env.connection, workload=env.workload)
    knob_state = env.action_space.get_knob_space().get_state(None)
    ql_knobs = env.action_space.get_knob_space().get_query_level_knobs(knob_state)

    changes = {}
    with open(autosteer_file) as f:
        for line in f:
            line = line.strip()
            parts = line.split(" ")
            assert len(parts) == 3
            changes[parts[0]] = parts[1]

    for query, hintset in changes.items():
        sets = [h.strip() for h in hintset.split(",")]
        sets = [s for s in sets if len(s) > 0]

        qid = query.split("/")[-1]
        qid = sql_mapping[qid]
        for setopt in sets:
            full_opt = f"{qid}_{setopt}"
            assert full_opt in ql_knobs, print(full_opt)
            ql_knobs[full_opt] = (ql_knobs[full_opt][0], 0)

    # Reboot and dump page cache.
    env._start_with_config_changes(conf_changes=None, dump_page_cache=True)
    collect_samples = []

    for sample_idx in range(num_samples):
        collect_samples.append(env.workload._execute_workload(
            connection=env.connection,
            workload_timeout=workload_timeout,
            ql_knobs=ql_knobs,
            env_spec=env.env_spec))

        if collect_samples[-1] == workload_timeout:
            break

    return collect_samples

def read_next_config(it, current):
    if "CHECKING" in current:
        config = eval(next(it).strip())
        return { k: (-1, v) for k, v in config.items() }, -1, None, True

    config = {}
    config_time = None
    while ("Initial input" not in current and "CHECKING" not in current) or len(config) == 0:
        current = next(it).strip()
        if "mythril" in current:
            if current.startswith("x"):
                ps = current.split(" ")
                assert ps[3] not in config
                config_time = float(ps[2])
                config[ps[3]] = (float(ps[4]), "(no hint)")

            else:
                ps = current.split(" ")
                config_time = float(ps[2])
                assert ps[3] not in config
                hintset = " ".join(ps[5:])
                if hintset == "None":
                    hintset = ""
                config[ps[3]] = (float(ps[4]), hintset)
    return config, config_time, current, False


def run_bao(env, samples, timeout, bao_file, benchmark_config_path, check_unit_sec, blocklist=[]):
    orig_timeout = timeout
    run_data = []
    current_step = 0

    start_time = 0
    num_baos = 0
    seed_config = {}
    with open(bao_file, "r") as f:
        for line in f:
            if line.startswith("x ") and start_time == 0:
                ps = line.split(" ")[2]
                start_time = float(ps)

            if line.startswith("x "):
                ps = line.split(" ")
                assert ps[3] not in seed_config
                seed_config[ps[3]] = (float(ps[4]), "(no hint)")

            if line.startswith("Initial input channels") or line.startswith("CHECKING CONFIG"):
                num_baos += 1

    with open(benchmark_config_path, "r") as f:
        qo = yaml.safe_load(f)["mythril"]["query_spec"]["query_order"]
    with open(qo, "r") as f:
        sql_mapping = {}
        for l in f:
            c = l.strip().split(",")
            sql_mapping[c[1]] = c[0]

    cache_runs = {}
    last_run = 0
    with tqdm.tqdm(total=num_baos) as pbar:
        with open(bao_file, "r") as f:
            it = iter(f)

            changes, config_time, next_line, is_last_run = read_next_config(it, "")
            while True:
                env.action_space.get_knob_space().reset(connection=env.connection, workload=env.workload)
                knob_state = env.action_space.get_knob_space().get_state(None)
                ql_knobs = env.action_space.get_knob_space().get_query_level_knobs(knob_state)

                # Continuously merge.
                for query, (reward, hintset) in changes.items():
                    if reward == -1 or reward <= seed_config[query][0]:
                        seed_config[query] = (reward, hintset)

                # But only run every 15 minutes at best...
                if config_time - last_run < check_unit_sec and (not is_last_run):
                    changes, config_time, next_line, is_last_run = read_next_config(it, next_line)
                    pbar.update(1)
                    continue

                for query, (_, hintset) in seed_config.items():
                    if hintset == "(no hint)":
                        continue

                    sets = [h.strip() for h in hintset.split(";")]
                    sets = [s for s in sets if len(s) > 0]

                    qid = query.split("/")[-1]
                    qid = sql_mapping[qid]
                    for setopt in sets:
                        opt = setopt.split("SET ")[-1].split(" TO")[0]
                        full_opt = f"{qid}_{opt}"
                        assert full_opt in ql_knobs, print(full_opt)
                        ql_knobs[full_opt] = (ql_knobs[full_opt][0], 0)

                cache_ql_knobs = {k: v for k, (_, v) in ql_knobs.items()}
                cache_key = json.dumps(cache_ql_knobs, sort_keys=True, cls=NpEncoder)
                if cache_key in cache_runs:
                    collect_samples = cache_runs[cache_key]

                else:
                    # Reboot and dump page cache.
                    env._start_with_config_changes(conf_changes=None, dump_page_cache=True)
                    logging.info(f"Running samples {current_step}")
                    collect_samples = []

                    query_runtime = {}
                    for sample_idx in range(samples):
                        collect_samples.append(env.workload._execute_workload(
                            connection=env.connection,
                            workload_timeout=timeout,
                            ql_knobs=ql_knobs,
                            env_spec=env.env_spec,
                            # Don't use pg_hint_plan.
                            disable_pg_hint=True,
                            blocklist=blocklist))

                        if collect_samples[-1] >= orig_timeout:
                            break

                        # Break samples.
                        if samples == 2 and sample_idx == 0 and collect_samples[-1] >= timeout:
                            break
                        elif samples > 2 and sample_idx == 1 and collect_samples[-1] >= timeout:
                            break

                    cache_runs[cache_key] = collect_samples

                # This is the new "timeout".
                if max(collect_samples) < timeout:
                    timeout = max(collect_samples)

                logging.info(f"Samples: {collect_samples}")

                data = {
                    "step": current_step,
                    "time_since_start": int(config_time - start_time),
                }
                data.update({f"runtime{i}": s for i, s in enumerate(collect_samples)})
                logging.debug(f"Collected samples: {collect_samples}")
                run_data.append(data)

                # Update "last_run".
                last_run = config_time

                current_step += 1
                pbar.update(1)
                if is_last_run:
                    # We are done!.
                    break

                changes, config_time, next_line, is_last_run = read_next_config(it, next_line)
            
            pd.DataFrame(run_data).to_csv("out.csv", index=False)
