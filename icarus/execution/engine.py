"""This module implements the simulation engine.

The simulation engine, given the parameters according to which a single
experiments needs to be run, instantiates all the required classes and executes
the experiment by iterating through the event provided by an event generator
and providing them to a strategy instance.
"""
import csv
import pickle, json
from pathlib import Path
import logging
from icarus.execution import (
    NetworkModel,
    NetworkView,
    NetworkController,
    CollectorProxy,
)
from icarus.registry import DATA_COLLECTOR, STRATEGY


__all__ = ["exec_experiment"]

logger = logging.getLogger("main")

def exec_experiment(topology, workload, netconf, strategy, cache_policy, collectors, period=None, save=False, curr_exp=None):
    """Execute the simulation of a specific scenario.

    Parameters
    ----------
    topology : Topology
        The FNSS Topology object modelling the network topology on which
        experiments are run.
    workload : iterable
        An iterable object whose elements are (time, event) tuples, where time
        is a float type indicating the timestamp of the event to be executed
        and event is a dictionary storing all the attributes of the event to
        execute
    netconf : dict
        Dictionary of attributes to inizialize the network model
    strategy : tree
        Strategy definition. It is tree describing the name of the strategy
        to use and a list of initialization attributes
    cache_policy : tree
        Cache policy definition. It is tree describing the name of the cache
        policy to use and a list of initialization attributes
    collectors: dict
        The collectors to be used. It is a dictionary in which keys are the
        names of collectors to use and values are dictionaries of attributes
        for the collector they refer to.

    Returns
    -------
    results : Tree
        A tree with the aggregated simulation results from all collectors
    """

    model = NetworkModel(topology, cache_policy, **netconf)
    view = NetworkView(model)
    controller = NetworkController(model)

    collectors_inst = [
        DATA_COLLECTOR[name](view, **params) for name, params in collectors.items()
    ]
    collector = CollectorProxy(view, collectors_inst)
    controller.attach_collector(collector)

    strategy_name = strategy["name"]  # "CL2SM"
    strategy_args = {k: v for k, v in strategy.items() if k != "name"}
    strategy_args["strategy_name"] = strategy_name  # ✅ add this
    strategy_inst = STRATEGY[strategy_name](view, controller, **strategy_args)
    
    # === RESTORE STRATEGY STATE (if previous period exists) ===
    if strategy_name == "CL2SM":
        strategy_inst.restore_strategy_state(strategy_name=strategy_name, period=period-1, curr_exp=curr_exp)

    # Specify the headers
    for i, (time, event) in enumerate(workload):
        logger.info("i: %s, time: %s, event: %s"%(i, time, event))
        strategy_inst.process_event(time, **event)
    
    if save and strategy_name == "CL2SM":
        state = {
            "gain_per_data": strategy_inst.gain_per_data,
            "request_counter": strategy_inst.request_counter,
        }

        try:
            saved_dir = Path("strategy_states")
            filepath = saved_dir / f"exp_{curr_exp}_{strategy_name}_p{period}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(state, f)
            print(f"[💾] Saved {strategy_name} state to {filepath}")

            jsonpath = filepath.with_suffix(".json")
            with open(jsonpath, "w") as jf:
                json.dump(state, jf, indent=2)
            print(f"[📄] JSON copy saved to {jsonpath}")
        except Exception as e:
            print(f"[⚠️] Failed to save {strategy_name} state (period {period}): {e}")

    model.collector_proxy = collector
    model.period = period
    return collector.results(), model
