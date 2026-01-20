"""Orchestrate the execution of all experiments.

The orchestrator is responsible for scheduling experiments specified in the
user-provided settings.
"""
import math
from pathlib import Path
import random
import time
import collections
import multiprocessing as mp
import logging
import copy
import sys
import signal
import traceback

from icarus.execution import exec_experiment
from icarus.extract_ci_per_node import load_nodes_ci
from icarus.registry import (
    TOPOLOGY_FACTORY,
    CACHE_PLACEMENT,
    CONTENT_PLACEMENT,
    CACHE_POLICY,
    WORKLOAD,
    DATA_COLLECTOR,
    STRATEGY,
)
from icarus.results import ResultSet
from icarus.util import SequenceNumber, Tree, timestr


__all__ = ["Orchestrator", "run_scenario"]


logger = logging.getLogger("orchestration")


class Orchestrator:
    """Orchestrator.

    It is responsible for orchestrating the execution of all experiments and
    aggregate results.
    """

    def __init__(self, settings, summary_freq=4):
        """Constructor

        Parameters
        ----------
        settings : Settings
            The settings of the simulator
        summary_freq : int
            Frequency (in number of experiment) at which summary messages
            are displayed
        """
        self.settings = settings
        self.results = ResultSet()
        self.seq = SequenceNumber()
        self.exp_durations = collections.deque(maxlen=30)
        self.n_success = 0
        self.n_fail = 0
        self.summary_freq = summary_freq
        self._stop = False
        if self.settings.PARALLEL_EXECUTION:
            self.pool = mp.Pool(settings.N_PROCESSES)

    def stop(self):
        """Stop the execution of the orchestrator"""
        logger.info("Orchestrator is stopping")
        self._stop = True
        if self.settings.PARALLEL_EXECUTION:
            self.pool.terminate()
            self.pool.join()

    def run(self):
        """Run the orchestrator.

        This call is blocking, whether multiple processes are used or not. This
        methods returns only after all experiments are executed.
        """
        # Create queue of experiment configurations
        queue = collections.deque(self.settings.EXPERIMENT_QUEUE)
        # Calculate number of experiments and number of processes
        self.n_exp = len(queue) * self.settings.N_REPLICATIONS
        self.n_proc = (
            self.settings.N_PROCESSES if self.settings.PARALLEL_EXECUTION else 1
        )
        logger.info(
            "Starting simulations: %d experiments, %d process(es)"
            % (self.n_exp, self.n_proc)
        )

        if self.settings.PARALLEL_EXECUTION:
            # Starting from Python 3.2, multiprocessing.Pool.apply_async
            # accepts a new error_callback argument that is a callable for
            # returning a message when uncaught errors are thrown.
            # The following lines ensure compatibility with Python < 3.2
            callbacks = {"callback": self.experiment_callback}
            if sys.version_info > (3, 2):
                callbacks["error_callback"] = self.error_callback
            # This job queue is used only to keep track of which jobs have
            # finished and which are still running. Currently this information
            # is used only to handle keyboard interrupts correctly
            job_queue = collections.deque()
            # Schedule experiments from the queue
            while queue:
                experiment = queue.popleft()
                for _ in range(self.settings.N_REPLICATIONS):
                    job_queue.append(
                        self.pool.apply_async(
                            run_scenario,
                            args=(
                                self.settings,
                                experiment,
                                self.seq.assign(),
                                self.n_exp,
                            ),
                            **callbacks
                        )
                    )
            self.pool.close()
            # This solution is probably not optimal, but at least makes
            # KeyboardInterrupt work fine, which is crucial if launching the
            # simulation remotely via screen.
            # What happens here is that we keep waiting for possible
            # KeyboardInterrupts till the last process terminates successfully.
            # We may have to wait up to 5 seconds after the last process
            # terminates before exiting, which is really negligible
            try:
                while job_queue:
                    job = job_queue.popleft()
                    while not job.ready():
                        time.sleep(5)
            except KeyboardInterrupt:
                self.pool.terminate()
            self.pool.join()

        else:  # Single-process execution
            while queue:
                experiment = queue.popleft()
                for _ in range(self.settings.N_REPLICATIONS):
                    # print(self.settings.__dict__)
                    self.experiment_callback(
                        run_scenario(
                            self.settings, experiment, self.seq.assign(), self.n_exp
                        )
                    )
                    if self._stop:
                        self.stop()

        logger.info(
            "END | Planned: %d, Completed: %d, Succeeded: %d, Failed: %d",
            self.n_exp,
            self.n_fail + self.n_success,
            self.n_success,
            self.n_fail,
        )

    def error_callback(self, msg):
        """Callback method called in case of error in Python > 3.2

        Parameters
        ----------
        msg : string
            Error message
        """
        logger.error("FAILURE | Experiment failed: {}".format(msg))
        self.n_fail += 1

    def experiment_callback(self, args):
        """Callback method called by run_scenario

        Parameters
        ----------
        args : tuple
            Tuple of arguments
        """
        # If args is None, that means that an exception was raised during the
        # execution of the experiment. In such case, ignore it
        if not args:
            self.n_fail += 1
            return
        # Extract parameters
        params, results, duration = args
        self.n_success += 1
        # Store results
        self.results.add(params, results)

        self.exp_durations.append(duration)
        if self.n_success % self.summary_freq == 0:
            # Number of experiments scheduled to be executed
            n_scheduled = self.n_exp - (self.n_fail + self.n_success)
            # Compute ETA
            n_cores = min(mp.cpu_count(), self.n_proc)
            mean_duration = sum(self.exp_durations) / len(self.exp_durations)
            eta = timestr(n_scheduled * mean_duration / n_cores, False)
            # Print summary
            logger.info(
                "SUMMARY | Completed: %d, Failed: %d, Scheduled: %d, ETA: %s",
                self.n_success,
                self.n_fail,
                n_scheduled,
                eta,
            )


def run_scenario(settings, params, curr_exp, n_exp):
    """Run a single scenario experiment

    Parameters
    ----------
    settings : Settings
        The simulator settings
    params : Tree
        experiment parameters tree
    curr_exp : int
        sequence number of the experiment
    n_exp : int
        Number of scheduled experiments

    Returns
    -------
    results : 3-tuple
        A (params, results, duration) 3-tuple. The first element is a dictionary
        which stores all the attributes of the experiment. The second element
        is a dictionary which stores the results. The third element is an
        integer expressing the wall-clock duration of the experiment (in
        seconds)
    """
    try:
        start_time = time.time()
        proc_name = mp.current_process().name
        logger = logging.getLogger("runner-%s" % proc_name)

        # Get list of metrics required
        def tree_to_dict(tree):
            if isinstance(tree, Tree):
                return {k: tree_to_dict(v) for k, v in tree.items()}
            elif isinstance(tree, list):
                return [tree_to_dict(v) for v in tree]
            else:
                return tree

        tree = copy.deepcopy(params)
        metrics = tree_to_dict(tree["data_collectors"]) if tree["data_collectors"] else settings.DATA_COLLECTORS
        cache_placement_tree = copy.deepcopy(params)

        # Set topology
        topology_spec = tree["topology"]
        topology_name = topology_spec.pop("name")
        if topology_name not in TOPOLOGY_FACTORY:
            logger.error(
                "No topology factory implementation for %s was found." % topology_name
            )
            return None
        topology = TOPOLOGY_FACTORY[topology_name](**topology_spec)
        
        # Set workload
        workload_spec = tree["workload"]
        workload_name = workload_spec.pop("name")
        if workload_name not in WORKLOAD:
            logger.error(
                "No workload implementation named %s was found." % workload_name
            )
            return None
        workload = WORKLOAD[workload_name](topology, **workload_spec)
        
        # Assign caches to nodes
        if "cache_placement" in tree:
            cachepl_spec = tree["cache_placement"]
            cachepl_name = cachepl_spec.pop("name")
            if cachepl_name not in CACHE_PLACEMENT:
                logger.error("No cache placement named %s was found." % cachepl_name)
                return None
            network_cache = cachepl_spec.pop("network_cache")
            cachepl_spec["cache_budget"] = workload.n_contents * network_cache
        
        # Assign cotennt to sources
        contpl_spec = tree["content_placement"]
        contpl_name = contpl_spec.pop("name")
        if contpl_name not in CONTENT_PLACEMENT:
            logger.error(
                "No content placement implementation named %s was found." % contpl_name
            )
            return None
        CONTENT_PLACEMENT[contpl_name](topology, workload.contents, **contpl_spec)

        # caching and routing strategy definition
        strategy = tree["strategy"]
        if strategy["name"] not in STRATEGY:
            logger.error(
                "No implementation of strategy %s was found." % strategy["name"]
            )
            return None

        # cache eviction policy definition
        cache_policy = tree["cache_policy"]
        if cache_policy["name"] not in CACHE_POLICY:
            logger.error(
                "No implementation of cache policy %s was found." % cache_policy["name"]
            )
            return None

        netconf = tree["netconf"]
        avg_content_size = getattr(workload, "avg_content_size", None)
        if avg_content_size is not None:
            netconf["avg_content_size"] = avg_content_size
        
        scenario = tree["desc"] if "desc" in tree else "Description N/A"
        logger.info(
            "Experiment %d/%d | Preparing scenario: %s", curr_exp, n_exp, scenario
        )
        
        print(f"Experiment {curr_exp}/{n_exp} | Preparing scenario: {scenario}")
        
        if any(m not in DATA_COLLECTOR for m in metrics):
            logger.error(
                "There are no implementations for at least one data collector specified"
            )
            return None
        
        collectors = metrics

        logger.info("Experiment %d/%d | Start simulation", curr_exp, n_exp)
        logger.info(f"Experiment {curr_exp}/{n_exp} | Start simulation")
        N_PERIODS = getattr(settings, "N_PERIODS", 3)
        prev_state_path = None
        save_state_path = None
        full_alloc_list = []
        folder = "/Users/lydia/Desktop/icarus/examples/lce-vs-probcache/carbon_intensities/carbon_profiles/"  # path containing all node CSVs
        day_str = "2024-05-24"
        nodes_ci = load_nodes_ci(folder, day_str, 24)
        
        if cachepl_name == "ALLOCATED":
            GREEN_PERIOD = getattr(settings, "GREEN_PERIOD")
            prev_state_path = getattr(settings, "PREV_STATE_PATH")
            print(f"green_period:{GREEN_PERIOD}")
            per = GREEN_PERIOD
        else:
            per = 0
        
        for node, data in topology.nodes(data=True):
            country = data.get("Country")
            if country is not None and country in nodes_ci:
                val = nodes_ci[country][per]
                if isinstance(val, float) and math.isnan(val):
                    ci_val = random.randint(50, 900)
                else:
                    ci_val = val
                topology.node[node]["carbon_intensity"] = ci_val
            else:
                topology.node[node]["carbon_intensity"] = random.randint(50,900)

        for period in range(N_PERIODS):
            logger.info(f"\n🚀 Running period {period + 1}/{N_PERIODS}")
            if period > 0:
                # Skip warmup for subsequent periods
                logger.info("[⏭️] Warmup skipped (resuming from saved network state)")
                workload_spec["n_warmup"] = 0
                workload_spec["seed"] = 1 + period
                cache_placement_workload_spec = cache_placement_tree["workload"]
                cache_placement_workload_spec["seed"] = 1 + period
                cache_placement_workload_spec["n_warmup"] = 0

                for node, data in topology.nodes(data=True):
                    country = data.get("Country")
                    if country is not None and country in nodes_ci:
                        val = nodes_ci[country][period]
                        if isinstance(val, float) and math.isnan(val):
                            # choose a fallback, e.g. random or regional average
                            ci_val = random.randint(50, 900)
                        else:
                            ci_val = val
                        topology.node[node]["carbon_intensity"] = ci_val
                    else:
                        topology.node[node]["carbon_intensity"] = random.randint(50,900)
            
            logger.info(f"cache allocation: {cachepl_name}")
            if "cache_placement" in tree:
                if cachepl_name == "GREEN":
                    CACHE_PLACEMENT[cachepl_name](topology, cachepl_spec["cache_budget"], cache_placement_tree=cache_placement_tree, metrics=metrics, settings=settings, allocs=full_alloc_list, max_evaluations=cachepl_spec["MAX_EVALUATION"], period=period, prev_state_path=prev_state_path)
                else:
                    CACHE_PLACEMENT[cachepl_name](topology, **cachepl_spec)
            
            workload = WORKLOAD[workload_name](topology, **workload_spec)
            full_alloc_list = []
            icr_candidates = topology.graph["icr_candidates"]
            for v in icr_candidates:
                stack = topology.node[v].get("stack", [])
                if len(stack) > 1 and "cache_size" in stack[1]:
                    full_alloc_list.append(stack[1]["cache_size"])
                else:
                    full_alloc_list.append(0)
            
            allocs = {}
            for v in topology.nodes():
                stack = topology.node[v].get("stack", [])
                if len(stack) > 1 and "cache_size" in stack[1]:
                    allocs[v] = stack[1]["cache_size"]
            cachepl_spec["cache_budget"] = sum(allocs.values())
            results = {"allocations": allocs}
            
            netconf["saved_state_file"] = prev_state_path
            save_state_path = f"exp{curr_exp}_{topology_name}_p{period + 1}"
            save = False if cachepl_name == "ALLOCATED" else True
            results, model = exec_experiment(
                topology, workload, netconf, strategy, cache_policy, collectors, save=save, prev_state_path=prev_state_path, save_state_path=save_state_path, cachepl_name=cachepl_name, green_period=per
            )
            
            try:
                if cachepl_name != "ALLOCATED":
                    if hasattr(model, "save_state"):
                        model.cache_placement = cachepl_name
                        model.network_cache = network_cache
                        model.alpha=workload_spec["alpha"]
                        model.period = period
                        model.save_state(filename_prefix=save_state_path)
                        prev_state_path = save_state_path
                        print(f"[💾] Saved network model state after Experiment {curr_exp} -> {save_state_path}")
                    else:
                        print(f"[⚠️] NetworkModel has no save_state() method.")
            except Exception as e:
                print(f"[❌] Failed to save network state: {e}")
        
        duration = time.time() - start_time
        logger.info(
            "Experiment %d/%d | End simulation | Duration %s.",
            curr_exp,
            n_exp,
            timestr(duration, True),
        )
        print(f"Experiment {curr_exp}/{n_exp} | End simulation | Duration {timestr(duration, True)}")
        
        return (params, results, duration)
    except KeyboardInterrupt:
        logger.error("Received keyboard interrupt. Terminating")
        sys.exit(-signal.SIGINT)
    except Exception as e:
        err_type = str(type(e)).split("'")[1].split(".")[1]
        err_message = e.message
        logger.error(
            "Experiment %d/%d | Failed | %s: %s\n%s",
            curr_exp,
            n_exp,
            err_type,
            err_message,
            traceback.format_exc(),
        )
