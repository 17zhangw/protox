import json
import psutil
import os
import shutil
import psycopg
import threading
from psycopg.errors import ProgramLimitExceeded
from psycopg.errors import QueryCanceled
from psycopg.errors import OperationalError
from psycopg.rows import dict_row
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict
from plumbum import local
from plumbum.commands.processes import ProcessTimedOut
from plumbum.commands.processes import ProcessExecutionError
from pathlib import Path
import logging

from envs.spec import Spec
from envs.repository import Repository
from envs.reward import RewardUtility
from envs.workload import Workload
from envs.spaces.utils import check_subspace, fetch_server_indexes, fetch_server_knobs, fetch_class
from utils.derive_repo_config import derive_repo_config


class PostgresEnv(gym.Env):
    # We don't support rendering.
    metadata = {"render_modes": []}
    # Specification.
    env_spec: Spec = None
    # Connection to the database server.
    connection: psycopg.Connection = None
    # Reward Utility.
    reward_utility: RewardUtility = None
    # Repository.
    repository: Repository = None
    # Workload.
    workload: Workload = None
    # Current state representation.
    current_state = None
    # Current step.
    current_step: int = 0
    # Whether we are executing an OLTP workload.
    oltp_workload: bool = False
    # Horizon for episode.
    horizon: int = None
    # Per-query Timeout.
    timeout: int = None
    # Baseline.
    baseline_state = None
    baseline_metric = None

    logger = None
    replay = False

    def save_state(self):
        return {
            "current_state": self.current_state,
            "current_step": self.current_step,
            "baseline_state": self.baseline_state,
            "baseline_metric": self.baseline_metric,
            "log_step": self.log_step,
        }

    def load_state(self, d):
        self.current_state = d["current_state"]
        self.current_step = d["current_step"]
        self.baseline_state = d["baseline_state"]
        self.baseline_metric = d["baseline_metric"]
        self.log_step = d["log_step"]

    def __init__(self,
        spec: Spec,
        horizon: int,
        timeout: int,
        reward_utility: RewardUtility,
        logger,
        replay=False):
        self.replay = replay

        self.logger = logger
        self.env_spec = spec
        self.action_space = spec.action_space
        self.observation_space = spec.observation_space
        self.workload = spec.workload
        self.horizon = horizon
        self.timeout = timeout
        self.reward_utility = reward_utility
        self.constraints = spec.constraints
        self.use_memory = self.constraints["memory_constraint"]

        # Construct repository.
        Path(spec.repository_path).mkdir(parents=True, exist_ok=True)
        self.repository = Repository(spec.repository_path, self.action_space)
        self.log_step = 0
        self.oltp_workload = spec.oltp_workload
        self.baseline_config = spec.baseline_config if hasattr(spec, "baseline_config") else None

    def _reward_constraints(self, success, results, update, qmetrics=None):
        metric, reward = None, None
        memviolate = False
        qviolate = False

        if success and (self.baseline_state is not None):
            # If not success, the DBMS may not actually be started up and running.
            memviolate = False if not self.use_memory else self._inject_memory()[0]
            if self.constraints["query_constraint"] and qmetrics:
                qabs = self.constraints["query_tolerance"].endswith("%")
                delta = float(self.constraints["query_tolerance"][:-1])

                for qid, (_, runtime, timeout) in qmetrics.items():
                    assert qid in self.baseline_relative_metric, print(qid)
                    base_runtime = self.baseline_relative_metric[qid][1]
                    base_runtime = base_runtime if base_runtime else (self.timeout * 1e6)
                    base_runtime = min(base_runtime, self.timeout * 1e6)

                    base_timeout = self.baseline_relative_metric[qid][2]
                    delta = (base_runtime * delta/100.) if qabs else (delta * 1e6)

                    if timeout and base_timeout:
                        # Don't penalize if both timeout since we don't know.
                        continue
                    elif timeout and (not base_timeout):
                        logging.debug(f"Q:{qid} timed out but base did not...")
                        qviolate = True
                        break
                    elif (not timeout) and base_timeout:
                        # No problem if we don't timeout and baseline timed out.
                        continue
                    elif (not timeout) and (runtime > base_runtime) and (runtime - base_runtime >= delta):
                        logging.debug(f"Q:{qid} {runtime} > {base_runtime} w. {delta} so qviolate.")
                        qviolate = True
                        break

        if self.reward_utility is not None:
            did_error = (not success) or memviolate or qviolate
            metric, reward = self.reward_utility(result_dir=results, update=update, did_error=did_error)

        return metric, reward, (memviolate or qviolate)

    def _inject_memory(self, state=None, error=False, old_bound=None):
        if self.use_memory:
            if not error:
                partial = self.constraints.get("partial_account", True)
                kc = self.action_space.get_knob_space().state_container
                qknob = self.action_space.get_knob_space().get_query_level_knobs(kc)
                exist_mem, mem = self.workload.compute_used_mem(self.connection, partial, kc, qknob)

                hard_limit = None
                if isinstance(self.constraints["memory_budget"], int) or isinstance(self.constraints["memory_budget"], float):
                    hard_limit = float(self.constraints["memory_budget"])
                elif self.constraints["memory_budget"].lower().endswith("gb"):
                    hard_limit = float(self.constraints["memory_budget"].lower().split("gb")[0])
                else:
                    assert self.constraints["memory_budget"].endswith("x")
                    mult = float(self.constraints["memory_budget"].split("x")[0])
                    hard_limit = exist_mem * mult

                bound = min(mem / hard_limit, 1.)
            else:
                assert old_bound is not None
                bound = old_bound

            if state is not None:
                logging.debug(f"Memory Constraint injecting {bound}.")
                state["memconstraint"] = np.array([bound], dtype=np.float32)
            return bound >= 1.0, bound
        return False, None

    def _start_with_config_changes(self, conf_changes=None, timeout=None, dump_page_cache=False, save_snapshot=False):
        start_time = time.time()
        if self.connection is not None:
            self.connection.close()
            self.connection = None

        # Install the new configuration changes.
        if conf_changes is not None:
            conf_changes.append("shared_preload_libraries='pg_hint_plan,hypopg'")

            with open(f"{self.env_spec.postgres_data}/postgresql.auto.conf", "w") as f:
                for change in conf_changes:
                    f.write(change)
                    f.write("\n")

        # Start postgres instance.
        self._shutdown_postgres()
        if Path(f"{self.env_spec.output_log_path}/pg.log").exists():
            shutil.move(f"{self.env_spec.output_log_path}/pg.log", f"{self.env_spec.output_log_path}/pg.log.{self.log_step}")
            self.log_step += 1

        if self.oltp_workload and save_snapshot and not self.replay and self.horizon > 1:
            # Create an archive of pgdata as a snapshot.
            local["tar"]["cf", f"{self.env_spec.postgres_data}.tgz.tmp", "-C", self.env_spec.postgres_path, self.env_spec.postgres_data_folder].run()

        # Make sure the PID lock file doesn't exist.
        pid_lock = Path(f"{self.env_spec.postgres_data}/postmaster.pid")
        assert not pid_lock.exists()

        if dump_page_cache:
            assert self.replay
            # Dump the OS page cache.
            os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')

        attempts = 0
        while not pid_lock.exists():
            # Try starting up.
            retcode, stdout, stderr = local[f"{self.env_spec.postgres_path}/pg_ctl"][
                "-D", self.env_spec.postgres_data,
                "--wait",
                "-t", "180",
                "-l", f"{self.env_spec.output_log_path}/pg.log",
                "start"].run(retcode=None)

            if retcode == 0 or pid_lock.exists():
                break

            logging.warn("startup encountered: (%s, %s)", stdout, stderr)
            attempts += 1
            if attempts >= 5:
                logging.error("Number of attempts to start postgres has exceeded limit.")
                assert False

        # Wait until postgres is ready to accept connections.
        num_cycles = 0
        while True:
            if timeout is not None and num_cycles >= timeout:
                # In this case, we've failed to start postgres.
                logging.error("Failed to start postgres before timeout...")
                return False

            retcode, _, _ = local[f"{self.env_spec.postgres_path}/pg_isready"][
                "--host", self.env_spec.postgres_host,
                "--port", str(self.env_spec.postgres_port),
                "--dbname", self.env_spec.postgres_db].run(retcode=None)
            if retcode == 0:
                break

            time.sleep(1)
            num_cycles += 1
            logging.debug("Waiting for postgres to bootup but it is not...")

        # Re-establish the connection.
        self.connection = psycopg.connect(self.env_spec.connection, autocommit=True, prepare_threshold=None)

        # Copy the temporary over since we know the temporary can load.
        if self.oltp_workload and save_snapshot and not self.replay and self.horizon > 1:
            shutil.move(f"{self.env_spec.postgres_data}.tgz.tmp", f"{self.env_spec.postgres_data}.tgz")

        return True

    def __execute_psql(self, sql):
        """Execute psql command."""

        low_sql = sql.lower()
        if "create index" in low_sql or "vacuum" in low_sql or "checkpoint" in low_sql:
            set_work_mem = True
        else:
            set_work_mem = False

        psql_conn = "postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
            user=self.env_spec.postgres_user,
            password=self.env_spec.postgres_password,
            host=self.env_spec.postgres_host,
            port=self.env_spec.postgres_port,
            dbname=self.env_spec.postgres_db)

        if set_work_mem:
            def cancel_fn(conn_str, conn):
                logging.info("CANCEL Function invoked!")
                with psycopg.connect(self.env_spec.connection, autocommit=True, prepare_threshold=None) as tconn:
                    r = [r for r in tconn.execute("SELECT pid FROM pg_stat_progress_create_index")]
                for row in r:
                    logging.info(f"Killing process {row[0]}")
                    try:
                        psutil.Process(row[0]).kill()
                    except:
                        pass
                logging.info("CANCEL Function finished!")

            # Start with config changes for index builds.
            cc = [
                "maintenance_work_mem = '4GB'",
                "max_worker_processes = 10",
                "max_parallel_workers = 10",
                "max_parallel_maintenance_workers = 10",
            ]
            assert self._start_with_config_changes(conf_changes=cc, timeout=self.env_spec.connect_timeout)

            with psycopg.connect(self.env_spec.connection, autocommit=True, prepare_threshold=None) as conn:
                conn.execute("SET maintenance_work_mem = '4GB'")
                conn.execute("SET statement_timeout = 300000")
                try:
                    timer = threading.Timer(300.0, cancel_fn, args=(self.env_spec.connection, conn))
                    timer.start()

                    conn.execute(sql)
                    timer.cancel()
                except ProgramLimitExceeded as e:
                    timer.cancel()
                    logging.debug(f"Action error: {e}")
                    return -1, None, str(e)
                except QueryCanceled as e:
                    timer.cancel()
                    logging.debug(f"Action error: {e}")
                    return -1, None, f"canceling statement: {sql}."
                except psycopg.OperationalError as e:
                    timer.cancel()
                    logging.debug(f"Action error: {e}")
                    return -1, None, f"canceling statement: {sql}."
                except psycopg.errors.UndefinedTable:
                    timer.cancel()
                    raise
            return 0, "", ""
        else:
            ret, stdout, stderr = local[f"{self.env_spec.postgres_path}/psql"][psql_conn, "--command", sql].run()
        return ret, stdout, stderr

    def _shutdown_postgres(self):
        """Shutds down postgres."""
        if not Path(self.env_spec.postgres_data).exists():
            return

        while True:
            logging.debug("Shutting down postgres...")
            _, stdout, stderr = local[f"{self.env_spec.postgres_path}/pg_ctl"][
                "stop",
                "--wait",
                "-t", "180",
                "-D", self.env_spec.postgres_data].run(retcode=None)
            time.sleep(1)
            logging.debug("Stop message: (%s, %s)", stdout, stderr)

            # Wait until pg_isready fails.
            retcode, _, _ = local[f"{self.env_spec.postgres_path}/pg_isready"][
                "--host", self.env_spec.postgres_host,
                "--port", str(self.env_spec.postgres_port),
                "--dbname", self.env_spec.postgres_db].run(retcode=None)

            exists = (Path(self.env_spec.postgres_data) / "postmaster.pid").exists()
            if not exists and retcode != 0:
                break

    def restore_pristine_snapshot(self, conf_changes=None, timeout=None):
        self._shutdown_postgres()
        # Remove the data directory and re-make it.
        local["rm"]["-rf", self.env_spec.postgres_data].run()
        local["mkdir"]["-m", "0700", "-p", self.env_spec.postgres_data].run()
        # Strip the "pgdata" so we can implant directly into the target postgres_data.
        local["tar"]["xf", self.env_spec.data_snapshot_path, "-C", self.env_spec.postgres_data, "--strip-components", "1"].run()
        # Imprint the required port.
        ((local["echo"][f"port={self.env_spec.postgres_port}"]) >> f"{self.env_spec.postgres_data}/postgresql.conf")()
        # Load and start the database.
        return self._start_with_config_changes(conf_changes=conf_changes, timeout=timeout)

    def _restore_last_snapshot(self):
        assert self.oltp_workload
        assert Path(f"{self.env_spec.postgres_data}.tgz").exists()
        self._shutdown_postgres()
        # Remove the data directory and re-make it.
        local["rm"]["-rf", self.env_spec.postgres_data].run()
        local["mkdir"]["-m", "0700", "-p", self.env_spec.postgres_data].run()
        local["tar"]["xf", f"{self.env_spec.postgres_data}.tgz", "-C", self.env_spec.postgres_data, "--strip-components", "1"].run()
        # Imprint the required port.
        ((local["echo"][f"port={self.env_spec.postgres_port}"]) >> f"{self.env_spec.postgres_data}/postgresql.conf")()

        success = self._start_with_config_changes(conf_changes=None, timeout=self.env_spec.connect_timeout)
        if success:
            knobs = fetch_server_knobs(
                self.connection,
                self.env_spec.tables,
                self.action_space.get_knob_space().knobs,
                workload=None)
            logging.debug(f"[Restored snapshot knobs]: {knobs}")

            _, indexes = fetch_server_indexes(self.connection, self.env_spec.tables)
            logging.debug(f"[Restored snapshot indexes]: {indexes}")

        return success

    def reset(self, seed=None, options=None):
        reset_start = time.time()
        logging.info("Resetting database system state to snapshot.")
        super().reset(seed=seed)
        metric = None if options is None else options.get("metric", None)
        state = None if options is None else options.get("state", None)
        config = None if options is None else options.get("config", None)
        accum_metric = None if options is None else options.get("accum_metric", None)
        load = False if options is None else options.get("load", False)
        reset_lsc = None if options is None else options.get("reset_lsc", None)

        self.current_step = 0
        info = {}

        if state is not None and config is not None and metric is not None:
            if self.oltp_workload and self.horizon == 1:
                # Restore a pristine snapshot of the world if OTLP and horizon = 1
                self.restore_pristine_snapshot()
            else:
                # Instead of restoring a pristine snapshot, just reset the knobs.
                # This in effect "resets" the baseline knob settings.
                self._start_with_config_changes(conf_changes=[], timeout=self.env_spec.connect_timeout, dump_page_cache=False)

            # Note that we do not actually update the baseline metric/reward used by the reward
            # utility. This is so the reward is not stochastic with respect to the starting state.
            # This also means the reward is deterministic w.r.t to improvement.
            self.current_state = state.copy()

            # We need to reset any internally tracked state first.
            self.action_space.reset(**{
                "connection": self.connection,
                "config": config,
                "workload": self.workload,
                "no_lsc": True,
            })

            args = {
                "benchbase_config_path": self.env_spec.benchbase_config_path,
                "original_benchbase_config_path": self.env_spec.original_benchbase_config_path,
                "load": load or (self.oltp_workload and self.horizon == 1),
            }
            # Maneuver the state into the requested state/config.
            config_changes, sql_commands = self.action_space.generate_plan_from_config(config, **args)
            self.shift_state(config_changes, sql_commands)

            if self.reward_utility is not None:
                self.reward_utility.set_relative_baseline(self.baseline_metric, prev_result=metric)

            self.current_state = state
            logging.debug("[Finished] Reset to state (config): %s", config)

        elif self.baseline_state is None:
            assert config is None
            # Restore a pristine snapshot of the world.
            self.restore_pristine_snapshot()

            if self.baseline_config is not None:
                logging.info(f"Loading from {self.baseline_config}")
                # Load the baseline configuration.
                use_template = False
                use_booster = False
                if "--template" in self.baseline_config:
                    use_template = True
                    self.baseline_config = self.baseline_config.replace("--template", "")
                if "--booster" in self.baseline_config:
                    use_booster = True
                    self.baseline_config = self.baseline_config.replace("--booster", "")

                assert Path(self.baseline_config).exists(), print(self.baseline_config)
                knobs, index_sqls = derive_repo_config(self, self.baseline_config, use_template, self.env_spec.benchmark, use_booster=use_booster)
                # Rename existing so we don't name conflict.
                index_sqls = [isq.replace(" eeindex", " eeeindex") for isq in index_sqls]
                index_sqls = [isq.replace(" eindex", " eeindex") for isq in index_sqls]
                index_sqls = [isq.replace(" index", " eindex") for isq in index_sqls]

                self.action_space.reset(**{
                    "connection": self.connection,
                    "config": None,
                    "workload": self.workload,
                    "no_lsc": True,
                })

                args = {
                    "benchbase_config_path": self.env_spec.benchbase_config_path,
                    "original_benchbase_config_path": self.env_spec.original_benchbase_config_path,
                    "load": load or (self.oltp_workload and self.horizon == 1),
                }

                # Maneuver the state into the requested state/config.
                cc, _ = self.action_space.get_knob_space().generate_plan(knobs, no_check=True)
                self.shift_state(cc, index_sqls, ignore_error=True)
                # Try to pipe the query level knob through.
                # Then try to pipe the config through to the reset() logic.
                config = (knobs, [])

            assert not self.replay

            # On the first time, run the benchmark to get the baseline.
            success, results, state, mutilated, _, accum_metric, _ = self.workload.execute(
                connection=self.connection,
                env_spec=self.env_spec,
                timeout=self.timeout,
                action=config,
                current_state=None,
            )

            # Get the metric.
            metric, _, _ = self._reward_constraints(success, results, False)
            logging.info(f"Baseline Benchmark iteration with metric {metric}")

            # Save the baseline run.
            local["mv"][results, f"{self.env_spec.repository_path}/baseline"].run()

            # Ensure that the first run succeeds.
            assert success

            if config is not None:
                # Propagate mutilated forwards...
                config = config if mutilated is None else mutilated
            else:
                # Ensure that the action is not mutilated since there is none!
                assert mutilated is None

            # Set the metric workload.
            self.env_spec.workload.set_workload_timeout(metric)

            self.reward_utility.set_relative_baseline(metric, prev_result=metric)
            _, reward = self.reward_utility(metric=metric, update=False, did_error=False)
            self.baseline_state = state
            self.baseline_metric = metric
            self.baseline_relative_metric = accum_metric
            self.current_state = self.baseline_state.copy()
            info = {"baseline_metric": metric, "baseline_reward": reward, "accum_metric": accum_metric}

        else:
            # Restore a pristine snapshot of the world.
            self.restore_pristine_snapshot()

            assert self.baseline_metric is not None
            self.current_state = self.baseline_state.copy()
            self.reward_utility.set_relative_baseline(self.baseline_metric, prev_result=self.baseline_metric)

        self.action_space.reset(**{
            "connection": self.connection,
            "config": config,
            "workload": self.workload,
            "reset_lsc": reset_lsc,
        })

        if self.env_spec.workload_eval_reset:
            target_state, target_metric = self.workload.reset(
                env_spec=self.env_spec,
                connection=self.connection,
                reward_utility=self.reward_utility,
                timeout=self.timeout,
                accum_metric=accum_metric)
        else:
            target_state, target_metric = self.workload.reset()
            assert target_state is None
            logging.debug("Reset without best observed evaluation.")

        if target_state is not None:
            # Implant the new target state.
            self.current_state = target_state.copy()
            if self.reward_utility is not None:
                self.reward_utility.set_relative_baseline(self.baseline_metric, prev_result=target_metric)

        if self.logger is not None:
            self.logger.record("instr_time/reset", int(time.time() - reset_start))

        # Set the correct current LSC.
        self.current_state["lsc"] = self.action_space.get_current_lsc()
        self._inject_memory(self.current_state)
        return self.current_state, info


    def step(self, action):
        assert not self.replay
        assert self.repository is not None
        q_timeout = False
        accum_metric = None
        truncated = False
        mutilated = None

        # Log the action in debug mode.
        logging.debug("[step]: %s", self.action_space.to_jsonable([action]))
        # Get the previous metric and penalty a-priori.
        previous_metric = self.reward_utility.previous_result
        worst_metric, worst_reward = self.reward_utility(did_error=True, update=False)

        # Get the memory.
        old_lsc = self.action_space.get_current_lsc()
        _, old_bound = self._inject_memory(state=None, error=False, old_bound=None)

        args = {
            "benchbase_config_path": self.env_spec.benchbase_config_path,
            "original_benchbase_config_path": self.env_spec.original_benchbase_config_path,
        }

        # Get the prior state.
        pre_indexes = []
        if self.action_space.get_index_space():
            pre_indexes = [ia.sql(add=True) for ia in self.action_space.get_index_space().state_container]
        old_state_container = {}
        if self.action_space.get_knob_space():
            old_state_container = self.action_space.get_knob_space().state_container.copy()

        # Save the old configuration file.
        old_conf_path = f"{self.env_spec.postgres_data}/postgresql.auto.conf"
        conf_path = f"{self.env_spec.postgres_data}/postgresql.auto.old"
        local["cp"][old_conf_path, conf_path].run()

        # Figure out what we have to change to get to the new configuration.
        config_changes, sql_commands = self.action_space.generate_action_plan(action, **args)
        # Attempt to maneuver to the new state.
        success = self.shift_state(config_changes, sql_commands)
        violate = False
        stattables = None

        if success:
            stattables = fetch_class(self.connection)

            # Evaluate the benchmark.
            start_time = time.time()
            success, results, next_state, mutilated, q_timeout, accum_metric, qhints = self.workload.execute(
                connection=self.connection,
                env_spec=self.env_spec,
                timeout=self.timeout,
                current_state=self.current_state.copy(),
                action=action,
            )

            consolidated = {
                "knobs": {
                    k: v
                    for k, v in fetch_server_knobs(
                        self.connection,
                        self.env_spec.tables,
                        self.action_space.get_knob_space().knobs,
                        workload=None,
                    ).items() if not k.startswith("Q")
                },
                "indexes": pre_indexes + [sc for sc in sql_commands if "index" in sc.lower()],
                "qknobs": qhints,
            }
            with open(f"{results}/config.json", "w") as f:
                json.dump(consolidated, f, indent=2)

            metric, reward, violate = self._reward_constraints(success, results, True, qmetrics=accum_metric)
            logging.info(f"Benchmark iteration with metric {metric} (reward: {reward}) (q_timeout: {q_timeout})")

            if self.logger is not None:
                self.logger.record("instr_time/workload_eval", time.time() - start_time)
        else:
            # Illegal configuration.
            logging.info("Found illegal configuration: %s. %s", config_changes, config_changes)
            self.action_space.flag_illegal(action, self.connection)
            violate = True

        if self.oltp_workload and self.horizon > 1:
            # If horizon = 1, then we're going to reset anyways. So easier to just untar the original archive.
            # Restore the crisp and clean snapshot.
            # If we've "failed" due to configuration, then we will boot up the last "bootable" version.
            self._restore_last_snapshot()

        if success:
            if not self.oltp_workload:
                # Update the workload metric timeout if we've succeeded.
                self.env_spec.workload.set_workload_timeout(metric)

            # Always incorporate the true state information to the repository.
            action = action if mutilated is None else mutilated
            _, repo_dir = self.repository.add(action, metric, reward, results, conf_path=conf_path, prior_state=(old_state_container, pre_indexes), stattables=stattables)
        else:
            # Since we reached an invalid area, just set the next state to be the current state.
            metric, reward = self.reward_utility(did_error=True)
            truncated = True
            next_state = self.current_state.copy()
            repo_dir = None

        if mutilated is not None:
            with torch.no_grad():
                # Extract the embedding version of the mutilated action now.
                # If we extract it later, then it get's shifted.
                mutilated = self.action_space.actor_action_embedding(mutilated)[0]

        self.current_step = self.current_step + 1
        self.current_state = next_state

        # Advance the action space tracking infrastructure.
        if not truncated:
            # Only advance the action space if we weren't truncated.
            self.action_space.advance(action, connection=self.connection, workload=self.workload)

        # Set the current LSC after advancing.
        self.current_state["lsc"] = self.action_space.get_current_lsc()
        # If truncated, the action failed to deploy.
        self._inject_memory(self.current_state, error=truncated, old_bound=old_bound)

        assert check_subspace(self.env_spec.observation_space, self.current_state)
        info = {
            "lsc": old_lsc.flatten(),
            "metric": metric,
            "mutilated_embed": mutilated,
            "q_timeout": q_timeout,
            "accum_metric": accum_metric,
            "violate": violate,
            "repo_dir": repo_dir,
        }

        if violate and self.constraints["reject"]:
            # Override the truncation flag if reject and we violate.
            # Do this here so the "accounting" has already been done...
            # Treat reject as just a normal "reset" path.
            truncated = True

        return self.current_state, reward, (self.current_step >= self.horizon), truncated, info

    def shift_state(self, config_changes, sql_commands, dump_page_cache=False, ignore_error=False):
        def attempt_checkpoint(conn_str):
            while True:
                try:
                    with psycopg.connect(conn_str, autocommit=True, prepare_threshold=None) as conn:
                        conn.execute("CHECKPOINT")
                    break
                except psycopg.OperationalError as e:
                    logging.debug(f"[attempt_checkpoint]: {e}")
                    time.sleep(5)

        shift_start = time.time()
        # First enforce the SQL command changes.
        for i, sql in enumerate(sql_commands):
            logging.info(f"Executing {sql} [{i+1}/{len(sql_commands)}]")
            try:
                ret, stdout, stderr = self.__execute_psql(sql)
                if not ignore_error:
                    if ret == -1:
                        print(stdout, stderr, flush=True)
                        assert "index row requires" in stderr or "canceling statement" in stderr
                        attempt_checkpoint(self.env_spec.connection)
                        return False

                    assert ret == 0, print(stdout, stderr)
            except ProcessExecutionError as e:
                if not ignore_error:
                    message = str(e)
                    print(message, flush=True)
                    assert "index row requires" in message or "canceling statement" in message
                    attempt_checkpoint(self.env_spec.connection)
                    return False
            except psycopg.errors.UndefinedTable as e:
                assert ignore_error

        # Now try and perform the configuration changes.
        ret = self._start_with_config_changes(
            conf_changes=config_changes,
            timeout=self.env_spec.connect_timeout,
            dump_page_cache=dump_page_cache,
            save_snapshot=True)

        if self.logger is not None:
            self.logger.record("instr_time/shift", int(time.time() - shift_start))
        return ret

    def close(self):
        self._shutdown_postgres()
        local["rm"]["-rf", self.env_spec.postgres_data].run()
        local["rm"]["-rf", f"{self.env_spec.postgres_data}.tgz"].run()
        local["rm"]["-rf", f"{self.env_spec.postgres_data}.tgz.tmp"].run()
