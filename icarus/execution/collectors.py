"""Performance metrics loggers

This module contains all data collectors that record events while simulations
are being executed and compute performance metrics.

Currently implemented data collectors allow users to measure cache hit ratio,
latency, path stretch and link load.

To create a new data collector, it is sufficient to create a new class
inheriting from the `DataCollector` class and override all required methods.
"""
import collections
import logging
import time

from icarus.registry import register_data_collector
from icarus.tools import cdf
from icarus.util import Tree, inheritdoc

logger = logging.getLogger("babel")


__all__ = [
    "DataCollector",
    "CollectorProxy",
    "CacheHitRatioCollector",
    "LinkLoadCollector",
    "LatencyCollector",
    "CostCollector",
    "CarbonFootprintCollector",
    "PathStretchCollector",
    "DummyCollector",
    "CHRCPCollector",
    "LiveReplicaMonitor",
]

chrcp = {}

class DataCollector:
    """Object collecting notifications about simulation events and measuring
    relevant metrics.
    """

    def __init__(self, view, **params):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            An instance of the network view
        params : keyworded parameters
            Collector parameters
        """
        self.view = view

    def start_session(self, timestamp, receiver, content, priority):
        """Notifies the collector that a new network session started.

        A session refers to the retrieval of a content from a receiver, from
        the issuing of a content request to the delivery of the content.

        Parameters
        ----------
        timestamp : int
            The timestamp of the event
        receiver : any hashable type
            The receiver node requesting a content
        content : any hashable type
            The content identifier requested by the receiver
        """
        pass

    def write_content(self, node, **kwargs):
        pass

    def cache_hit(self, node, **kwargs):
        """Reports that the requested content has been served by the cache at
        node *node*.

        Parameters
        ----------
        node : any hashable type
            The node whose cache served the content
        """
        pass

    def cache_miss(self, node):
        """Reports that the cache at node *node* has been looked up for
        requested content but there was a cache miss.

        Parameters
        ----------
        node : any hashable type
            The node whose cache served the content
        """
        pass

    def server_hit(self, node, **kwargs):
        """Reports that the requested content has been served by the server at
        node *node*.

        Parameters
        ----------
        node : any hashable type
            The server node which served the content
        """
        pass

    def request_hop(self, u, v, **kwargs):
        """Reports that a request has traversed the link *(u, v)*

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that link link is on the main path that will
            lead to hit a content. It is normally used to calculate latency
            correctly in multicast cases. Default value is *True*
        """
        pass

    def content_hop(self, u, v, **kwargs):
        """Reports that a content has traversed the link *(u, v)*

        Parameters
        ----------
        u : any hashable type
            Origin node
        v : any hashable type
            Destination node
        main_path : bool, optional
            If *True*, indicates that this link is being traversed by content
            that will be delivered to the receiver. This is needed to
            calculate latency correctly in multicast cases. Default value is
            *True*
        """
        pass

    def end_session(self, success=True):
        """Reports that the session is closed, i.e. the content has been
        successfully delivered to the receiver or a failure blocked the
        execution of the request

        Parameters
        ----------
        success : bool, optional
            *True* if the session was completed successfully, *False* otherwise
        """
        pass

    def storage_div(self, path, storage_gain, storage_loss, dep, stor, band, trans, pen, min_gain):
        pass

    def results(self):
        """Returns the aggregated results measured by the collector.

        Returns
        -------
        results : dict
            Dictionary mapping metric with results.
        """
        pass


# Note: The implementation of CollectorProxy could be improved to avoid having
# to rewrite almost identical methods, for example by playing with __dict__
# attribute. However, it was implemented this way to make it more readable and
# easier to understand.
class CollectorProxy(DataCollector):
    """This class acts as a proxy for all concrete collectors towards the
    network controller.

    An instance of this class registers itself with the network controller and
    it receives notifications for all events. This class is responsible for
    dispatching events of interests to concrete collectors.
    """

    EVENTS = (
        "start_session",
        "end_session",
        "write_content",
        "cache_hit",
        "cache_miss",
        "server_hit",
        "request_hop",
        "content_hop",
        "storage_div",
        "results",
    )

    def __init__(self, view, collectors):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            An instance of the network view
        collector : list of DataCollector
            List of instances of DataCollector that will be notified of events
        """
        self.view = view
        self.collectors = {
            e: [c for c in collectors if e in type(c).__dict__] for e in self.EVENTS
        }

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        for c in self.collectors["start_session"]:
            c.start_session(timestamp, receiver, content, priority)

    @inheritdoc(DataCollector)
    def write_content(self, node, **kwargs):
        for c in self.collectors["write_content"]:
            c.write_content(node, **kwargs)

    @inheritdoc(DataCollector)
    def cache_hit(self, node, **kwargs):
        for c in self.collectors["cache_hit"]:
            c.cache_hit(node, **kwargs)

    @inheritdoc(DataCollector)
    def cache_miss(self, node):
        for c in self.collectors["cache_miss"]:
            c.cache_miss(node)

    @inheritdoc(DataCollector)
    def server_hit(self, node, **kwargs):
        for c in self.collectors["server_hit"]:
            c.server_hit(node, **kwargs)

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, **kwargs):
        for c in self.collectors["request_hop"]:
            c.request_hop(u, v, **kwargs)

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, **kwargs):
        for c in self.collectors["content_hop"]:
            c.content_hop(u, v, **kwargs)

    @inheritdoc(DataCollector)
    def storage_div(self, path, storage_gain, storage_loss, dep, stor, band, trans, pen, min_gain):
        for c in self.collectors["storage_div"]:
            c.storage_div(path, storage_gain, storage_loss, dep, stor, band, trans, pen, min_gain)

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        for c in self.collectors["end_session"]:
            c.end_session(success)

    @inheritdoc(DataCollector)
    def results(self):
        return Tree(**{c.name: c.results() for c in self.collectors["results"]})

@register_data_collector("LINK_LOAD")
class LinkLoadCollector(DataCollector):
    """Data collector measuring the link load"""

    def __init__(self, view, req_size=150, content_size=1500):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        req_size : int
            Average size (in bytes) of a request
        content_size : int
            Average size (in byte) of a content
        """
        self.view = view
        self.req_count = collections.defaultdict(int)
        self.cont_count = collections.defaultdict(int)
        if req_size <= 0 or content_size <= 0:
            raise ValueError("req_size and content_size must be positive")
        self.req_size = req_size
        self.content_size = content_size
        self.t_start = -1
        self.t_end = 1

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        if self.t_start < 0:
            self.t_start = timestamp
        self.t_end = timestamp

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, **kwargs):
        self.req_count[(u, v)] += 1

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, **kwargs):
        self.cont_count[(u, v)] += 1

    @inheritdoc(DataCollector)
    def results(self):
        duration = self.t_end - self.t_start
        used_links = set(self.req_count.keys()).union(set(self.cont_count.keys()))
        link_loads = {
            link: (
                self.req_size * self.req_count[link]
                + self.content_size * self.cont_count[link]
            )
            / duration
            for link in used_links
        }
        link_loads_int = {
            link: load
            for link, load in link_loads.items()
            if self.view.link_type(*link) == "internal"
        }
        link_loads_ext = {
            link: load
            for link, load in link_loads.items()
            if self.view.link_type(*link) == "external"
        }
        mean_load_int = (
            sum(link_loads_int.values()) / len(link_loads_int)
            if len(link_loads_int) > 0
            else 0
        )
        mean_load_ext = (
            sum(link_loads_ext.values()) / len(link_loads_ext)
            if len(link_loads_ext) > 0
            else 0
        )
        return Tree(
            {
                "MEAN_INTERNAL": mean_load_int,
                "MEAN_EXTERNAL": mean_load_ext,
                "PER_LINK_INTERNAL": link_loads_int,
                "PER_LINK_EXTERNAL": link_loads_ext,
            }
        )

@register_data_collector("LATENCY")
class LatencyCollector(DataCollector):
    """Data collector measuring latency, i.e. the delay taken to delivery a
    content.
    """

    def __init__(self, view, cdf=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        cdf : bool, optional
            If *True*, also collects a cdf of the latency
        """
        self.cdf = cdf
        self.view = view
        self.req_latency = 0.0
        self.sess_count = 0
        self.latency = 0.0
        if cdf:
            self.latency_data = collections.deque()

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        self.sess_count += 1
        self.sess_latency = 0.0

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, **kwargs):
        main_path = kwargs.get("main_path") or True
        if main_path:
            self.sess_latency += self.view.link_delay(u, v)

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, **kwargs):
        main_path = kwargs.get("main_path") or True
        if main_path:
            self.sess_latency += self.view.link_delay(u, v)

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        if not success:
            return
        if self.cdf:
            self.latency_data.append(self.sess_latency)
        self.latency += self.sess_latency
        
        # logger.info(f"LatencyCollector latency:{self.sess_latency}")

    @inheritdoc(DataCollector)
    def results(self):
        results = Tree({"MEAN": self.latency / self.sess_count})
        if self.cdf:
            results["CDF"] = cdf(self.latency_data)
        return results

@register_data_collector("COST")
class CostCollector(DataCollector):
    """Data collector measuring cost, i.e. the penalty from delivering a
    content.
    """

    def __init__(self, view, **params):
        """Constructor

        Parameters
        ----------
        view : NetworkView
        The network view instance
        params : cost model and tiers info
        """
        
        self.request_size = 150 * 8
        self.sess_depreciation_cost = 0.0
        self.sess_bandwidth_cost = 0.0
        self.sess_read_cost = 0.0
        self.sess_write_cost = 0.0
        self.sess_routers_energy_cost = 0.0
        self.sess_links_energy_cost = 0.0
        self.sess_penalty_cost = 0.0
        self.sess_cost = 0.0

        self.depreciation_cost = 0.0
        self.bandwidth_cost = 0.0
        self.read_cost = 0.0
        self.write_cost = 0.0
        self.routers_energy_cost = 0.0
        self.links_energy_cost = 0.0
        self.penalty_cost = 0.0
        self.cost = 0.0

        self.view = view
        self.sess_count = 0

        self.cost_params = params['cost_params']

        self.penalty_table = self.cost_params ['penalty_table']
        self.cost_per_joule = self.cost_params ['cost_per_joule']
        self.cost_per_bit = self.cost_params ['cost_per_bit']
        self.router_energy_density = self.cost_params ['router_energy_density']
        self.link_energy_density = self.cost_params ['link_energy_density']

        self.log_file_path = 'path_log.csv'

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        self.sess_count += 1
        self.sess_latency = 0.0
        self.content = content
        self.receiver = receiver
        self.priority = priority
        self.sess_depreciation_cost = 0.0
        self.sess_bandwidth_cost = 0.0
        self.sess_read_cost = 0.0
        self.sess_write_cost = 0.0
        self.sess_routers_energy_cost = 0.0
        self.sess_links_energy_cost = 0.0
        self.sess_penalty_cost = 0.0
        self.sess_cost = 0.0

    def _resolve_cache_tiers(self, node, cache_tiers_kw):
        # Prefer the list passed in the event kwargs
        if isinstance(cache_tiers_kw, list):
            return cache_tiers_kw
        # Fallback to the view (per-node tiers built in NetworkModel)
        try:
            return self.view.cache_tiers().get(node, [])
        except Exception:
            return getattr(self.view, "model", None) and getattr(self.view.model, "node_tiers", {}).get(node, []) or []

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, **kwargs):
        main_path = kwargs.get("main_path") or True
        if main_path:
            self.sess_latency += self.view.link_delay(u, v)
            self.sess_routers_energy_cost += self.request_size * self.router_energy_density * self.cost_per_joule
            self.sess_links_energy_cost += self.request_size * self.link_energy_density * self.cost_per_joule
            self.sess_bandwidth_cost += self.request_size * self.cost_per_bit
            
    @inheritdoc(DataCollector)
    def content_hop(self, u, v, **kwargs):
        main_path = kwargs.get("main_path") or True
        if main_path:
            content_size = kwargs["size"] * 8
            self.sess_latency += self.view.link_delay(u, v)
            self.sess_routers_energy_cost += content_size * self.router_energy_density * self.cost_per_joule
            self.sess_links_energy_cost += content_size * self.link_energy_density * self.cost_per_joule
            self.sess_bandwidth_cost += content_size * self.cost_per_bit
            
    @inheritdoc(DataCollector)
    def cache_hit(self, node, **kwargs):
        # read cost
        content_size = kwargs["size"]
        tier_index = kwargs.get("tier_index")
        cache_tiers = self._resolve_cache_tiers(node, kwargs.get("cache_tiers"))
        if not cache_tiers:
            return
        
        if tier_index is None or not (0 <= tier_index < len(cache_tiers)):
            tier_index = 0
            
        tier = cache_tiers[tier_index]
        tier_active_power_density  = tier['active_caching_power_density']
        tier_idle_power_density = tier['idle_power_density']

        tiers_last_access = self.view.get_last_access(node)
        
        idle_time = 0.0 if tiers_last_access[tier_index]==0 else time.time() - tiers_last_access[tier_index]
        read_time = tier['latency'] + content_size / tier['read_throughput']
        self.sess_read_cost += ((tier_idle_power_density * idle_time) + (tier_active_power_density * content_size * 8 * read_time)) * self.cost_per_joule
        
        # depreciation cost
        if tier_index != 0:
            tier_max_capacity = tier['actual_size_bytes']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            self.sess_depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)

    @inheritdoc(DataCollector)
    def server_hit(self, node, **kwargs):
        content_size = kwargs["size"]
        server_latency = 1e-7
        server_read_throughput = 4e+10
        server_active_power_density = 10**-9
        read_time = server_latency + content_size / server_read_throughput
        self.sess_read_cost +=  server_active_power_density * read_time * content_size * 8 * self.cost_per_joule
        
        server_size = kwargs.get("server_size")
        server_purchase_cost = 200
        server_lifespan = 5 * 365 * 24 * 60 * 60
        self.sess_depreciation_cost += (content_size * server_purchase_cost) / (server_lifespan * server_size)
        
    @inheritdoc(DataCollector)
    def write_content(self, node, **kwargs):
        content_size = kwargs["size"]
        tier_index = kwargs.get("tier_index")
        
        cache_tiers = self._resolve_cache_tiers(node, kwargs.get("cache_tiers"))
        if not cache_tiers:
            return

        if tier_index is None or not (0 <= tier_index < len(cache_tiers)):
            tier_index = 0
        tiers_last_access = self.view.get_last_access(node)
        for i, tier in enumerate(cache_tiers[tier_index:], start=tier_index):
            tier_active_power_density  = tier['active_caching_power_density']
            tier_idle_power_density = tier['idle_power_density']
            
            idle_time = 0.0 if tiers_last_access[i]==0 else time.time() - tiers_last_access[i]
            write_time = tier['latency'] + content_size / tier['write_throughput']
            self.sess_write_cost += ((tier_idle_power_density * idle_time) + (tier_active_power_density * write_time * content_size * 8)) * self.cost_per_joule
            
            tier_max_capacity = tier['actual_size_bytes']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            self.sess_depreciation_cost += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        if not success:
            logger.info(f"end failed session")
            return
        self.depreciation_cost += self.sess_depreciation_cost
        self.bandwidth_cost += self.sess_bandwidth_cost
        self.read_cost += self.sess_read_cost
        self.write_cost += self.sess_write_cost
        self.routers_energy_cost += self.sess_routers_energy_cost
        self.links_energy_cost += self.sess_links_energy_cost
        for entry in self.penalty_table:
            if self.sess_latency <= entry["delay"]:
                if self.priority == "high":
                    self.sess_penalty_cost += entry["P0"] * 1e-8
                    self.penalty_cost += self.sess_penalty_cost
                    break
                elif self.priority == "low":
                    self.sess_penalty_cost += entry["P1"] * 1e-8
                    self.penalty_cost += self.sess_penalty_cost
                    break
        self.sess_cost += self.sess_bandwidth_cost + self.sess_routers_energy_cost + self.sess_links_energy_cost + self.sess_penalty_cost + self.sess_depreciation_cost + self.sess_read_cost + self.sess_write_cost
        self.cost += self.sess_cost
        # logger.info(f"end_session. cost : {self.cost}")
        # logger.info(f"end_session. sess_cost : {self.sess_cost}")
        # logger.info(f"end_session. sess_penalty_cost : {self.sess_penalty_cost}") 
        # logger.info(f"end_session. bandwidth_cost:{self.bandwidth_cost}, read_cost : {self.read_cost}, write_cost:{self.write_cost}, routers_energy_cost:{self.routers_energy_cost}, links_energy_cost:{self.links_energy_cost}")
        logger.info(f"REAL {self.sess_count}, {self.receiver}, {self.content}, {self.sess_bandwidth_cost}, {self.sess_routers_energy_cost + self.sess_links_energy_cost}, {self.sess_penalty_cost}, {self.sess_depreciation_cost}, {self.sess_read_cost+ self.sess_write_cost}")
        # Log the session costs to a file
                 # Log the session costs to a CSV file
        # with open(self.log_file_path, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([self.sess_count, self.receiver, self.content, self.sess_bandwidth_cost,
        #                      self.sess_routers_energy_cost+ self.sess_links_energy_cost, self.sess_penalty_cost,
        #                      self.sess_depreciation_cost, self.sess_read_cost+ self.sess_write_cost])

    @inheritdoc(DataCollector)
    def results(self):
        results = Tree(
            {
            "MEAN": self.cost / self.sess_count,
            "DEPRECIATION": self.depreciation_cost / self.sess_count,
            "BANDWIDTH": self.bandwidth_cost / self.sess_count,
            "READ_STORAGE": self.read_cost / self.sess_count,
            "WRITE_STORAGE": self.write_cost / self.sess_count,
            "ROUTERS": self.routers_energy_cost / self.sess_count,
            "LINKS": self.links_energy_cost / self.sess_count,
            "PENALTY": self.penalty_cost / self.sess_count
            })
        chrcp["cost"] = round(self.cost, 3)
        return results

@register_data_collector("CHRCP")
class CHRCPCollector(DataCollector):
    """Data collector measuring cost, i.e. the penalty from delivering a
    content.
    """

    def __init__(self, view, **params):
        """Constructor

        Parameters
        ----------
        view : NetworkView
        The network view instance
        params : cost model and tiers info
        """
        self.view = view

    @inheritdoc(DataCollector)
    def results(self):
        results = Tree(
            {
            "MEAN": chrcp["cost"]/chrcp["chr"] if chrcp["chr"] != 0 else 0
            })
        return results

@register_data_collector("CARBONFOOTPRINT")
class CarbonFootprintCollector(DataCollector):
    """Data collector measuring Carbon footprint
    """

    def __init__(self, view, **params):
        """Constructor

        Parameters
        ----------
        view : NetworkView
        The network view instance
        params : carbon footprint model and tiers info
        """
        
        self.request_size = 150 * 8 # bytes -> bits
        self.sess_depreciation_cf = 0.0
        self.sess_read_cf = 0.0
        self.sess_write_cf = 0.0
        self.sess_idle_cf = 0.0
        self.sess_routers_energy_cf = 0.0
        self.sess_links_energy_cf = 0.0
        self.sess_cf = 0.0
        self.sess_embodied_cf = 0.0

        self.depreciation_cf = 0.0
        self.read_cf = 0.0
        self.write_cf = 0.0
        self.idle_cf = 0.0
        self.routers_energy_cf = 0.0
        self.links_energy_cf = 0.0
        self.cf = 0.0
        self.embodied_cf = 0.0

        self.view = view
        self.sess_count = 0

        self.cost_params = params['cost_params']
        self.router_energy_density = self.cost_params ['router_energy_density'] # j/bit
        self.link_energy_density = self.cost_params ['link_energy_density'] # j/bit

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        self.sess_count += 1
        self.content = content
        self.receiver = receiver
        self.priority = priority
        self.timestamp = timestamp

        self.sess_depreciation_cf = 0.0
        self.sess_read_cf = 0.0
        self.sess_write_cf = 0.0
        self.sess_routers_energy_cf = 0.0
        self.sess_links_energy_cf = 0.0
        self.sess_idle_cf = 0.0
        self.sess_cf = 0.0
        self.sess_embodied_cf = 0.0

    def _resolve_cache_tiers(self, node, cache_tiers_kw):
        if isinstance(cache_tiers_kw, list):
            return cache_tiers_kw
        try:
            return self.view.cache_tiers().get(node, [])
        except Exception:
            return getattr(self.view, "model", None) and getattr(self.view.model, "node_tiers", {}).get(node, []) or []

    def _ci(self, node_hint=None, ci_kw=None):
        if ci_kw is not None:
            return ci_kw
        # fallback: use node carbon intensity if available, else 0.3
        model = getattr(self.view, "model", None)
        if model is not None and hasattr(model, "node_carbon_intensity") and node_hint is not None:
            return model.node_carbon_intensity.get(node_hint, 0.3)
        return 0.3
    
    @inheritdoc(DataCollector)
    def request_hop(self, u, v, **kwargs):
        main_path = kwargs.get("main_path") or True
        if main_path:
            ci = kwargs.get("carbon_intensity")
            if ci is None:  # fallback: approximate with source node CI
                ci = self._ci(node_hint=u)
            self.sess_routers_energy_cf += self.request_size * self.router_energy_density * ci / 3.6e6
            self.sess_links_energy_cf += self.request_size * self.link_energy_density * ci / 3.6e6
            
    @inheritdoc(DataCollector)
    def content_hop(self, u, v, **kwargs):
        main_path = kwargs.get("main_path") or True
        if main_path:
            ci = kwargs.get("carbon_intensity")
            if ci is None:
                ci = self._ci(node_hint=u)  # Kg CO2 eq/KwH
            content_size = kwargs["size"] * 8  # bytes -> bit
            self.sess_routers_energy_cf += content_size * self.router_energy_density * ci / 3.6e6   # Kg CO2 eq
            self.sess_links_energy_cf += content_size * self.link_energy_density * ci / 3.6e6   # Kg CO2 eq
            
    @inheritdoc(DataCollector)
    def cache_hit(self, node, **kwargs):
        # read cf
        content_size = kwargs["size"]  # Byte
        tier_index = kwargs.get("tier_index") or 0
        cache_tiers = self._resolve_cache_tiers(node, kwargs.get("cache_tiers"))
        if not cache_tiers:
            return
        if tier_index is None or not (0 <= tier_index < len(cache_tiers)):
            tier_index = 0
            
        ci = self._ci(node_hint=node, ci_kw=kwargs.get("carbon_intensity")) # Kg CO2 eq/KwH
        tier  = cache_tiers[0]
        
        tier_idle_power_density = tier['idle_power_density']    # w
        tiers_last_access = self.view.get_last_access(node)
        idle_time = 0.0 if tiers_last_access[tier_index]==0 else time.time() - tiers_last_access[tier_index]    # s
        self.sess_idle_cf += tier_idle_power_density * idle_time * ci / 3.6e6
        
        tier_active_power_density  = tier['active_caching_power_density']   # w/bit
        read_time = tier['latency'] + content_size / tier['read_throughput']    # s
        self.sess_read_cf += tier_active_power_density * content_size * 8 * read_time * ci / 3.6e6
        
        tier_max_capacity_bytes = tier['actual_size_bytes']
        
        # depreciation cf
        if tier_index != 0:
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            self.sess_depreciation_cf += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity_bytes)
            
        # Embodied CF amortization
        TE = tier["embodied_kgco2e_per_gb"] * tier_max_capacity_bytes / (8 * 1024**3)
        EL = tier["lifespan"] * 365 * 24 * 60 * 60  # lifespan in seconds
        RR = content_size / (8 * 1024**3)  # content size in GB
        TR = tier_max_capacity_bytes
        ecf = TE * (read_time / EL) * (RR / TR)
        self.sess_embodied_cf += ecf

    @inheritdoc(DataCollector)
    def server_hit(self, node, **kwargs):
        ci = self._ci(node_hint=node, ci_kw=kwargs.get("carbon_intensity"))
        content_size = kwargs["size"]
        server_latency = 1e-7
        server_read_throughput = 4e+10
        server_active_power_density = 10**-9
        read_time = server_latency + content_size / server_read_throughput
        self.sess_read_cf +=  server_active_power_density * read_time * content_size * 8 * ci
        
        server_size = kwargs.get("server_size")
        server_purchase_cost = 200
        server_lifespan = 5 * 365 * 24 * 60 * 60
        self.sess_depreciation_cf += (content_size * server_purchase_cost) / (server_lifespan * server_size)
        server_capacity_gb = server_size / (8 * 1024**3)
        TE = 0.2 * server_capacity_gb
        RR = content_size / (8 * 1024**3)
        ecf = TE * (read_time / server_lifespan) * (RR / server_capacity_gb)
        self.sess_embodied_cf += ecf
        
    @inheritdoc(DataCollector)
    def write_content(self, node, **kwargs):
        content_size = kwargs["size"]
        tier_index = kwargs.get("tier_index") or 0
        cache_tiers = self._resolve_cache_tiers(node, kwargs.get("cache_tiers"))
        if not cache_tiers:
            return
        if tier_index is None or not (0 <= tier_index < len(cache_tiers)):
            tier_index = 0

        ci = self._ci(node_hint=node, ci_kw=kwargs.get("carbon_intensity"))

        for i, tier in enumerate(cache_tiers[tier_index:], start=tier_index):
            tier_idle_power_density = tier['idle_power_density']
            tiers_last_access = self.view.get_last_access(node)
            idle_time = 0.0 if tiers_last_access[i]==0 else time.time() - tiers_last_access[i]
            self.sess_idle_cf += tier_idle_power_density * idle_time  * ci / 3.6e6
            
            tier_active_power_density  = tier['active_caching_power_density']
            write_time = tier['latency'] + content_size / tier['write_throughput']
            self.sess_write_cf += tier_active_power_density * write_time * content_size * 8 * ci / 3.6e6
            
            tier_max_capacity = tier['actual_size_bytes']
            tier_purchase_cost = tier['purchase_cost']
            tier_lifespan = tier['lifespan'] * 365 * 24 * 60 * 60
            self.sess_depreciation_cf += (content_size * tier_purchase_cost) / (tier_lifespan * tier_max_capacity)
            
            # --- Embodied CF calculation ---
            tier_capacity_gb = tier_max_capacity / (8 * 1024**3)
            TE = tier["embodied_kgco2e_per_gb"] * tier_capacity_gb
            RR = content_size / (8 * 1024**3)
            TR = tier_capacity_gb
            ecf = TE * (write_time / tier_lifespan) * (RR / TR)
            self.sess_embodied_cf += ecf

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        if not success:
            logger.info(f"end failed session")
            return
        self.depreciation_cf += self.sess_depreciation_cf
        self.read_cf += self.sess_read_cf
        self.write_cf += self.sess_write_cf
        self.routers_energy_cf += self.sess_routers_energy_cf
        self.links_energy_cf += self.sess_links_energy_cf
        self.idle_cf += self.sess_idle_cf
        self.sess_cf += self.sess_idle_cf + self.sess_depreciation_cf + self.sess_read_cf + self.sess_write_cf
        self.cf += self.sess_cf
        self.embodied_cf += self.sess_embodied_cf
        # logger.info(f"end_session. carbon footprint : {self.cf}")
        # logger.info(f"end_session. sess_cf : {self.sess_cf}")
        # logger.info(f"end_session. read_cf : {self.read_cf}, write_cf:{self.write_cf}, routers_energy_cf:{self.routers_energy_cf}, links_energy_cf:{self.links_energy_cf}")
        logger.info(f"REAL {self.sess_count}, {self.receiver}, {self.content}, {self.sess_routers_energy_cf + self.sess_links_energy_cf}, {self.sess_depreciation_cf}, {self.sess_read_cf+ self.sess_write_cf}")
        # Log the session cfs to a file
                 # Log the session cfs to a CSV file
        # with open(self.log_file_path, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([self.sess_count, self.receiver, self.content,
        #                      self.sess_routers_energy_cf+ self.sess_links_energy_cf,
        #                      self.sess_depreciation_cf, self.sess_read_cf+ self.sess_write_cf])

    @inheritdoc(DataCollector)
    def results(self):
        results = Tree(
            {
            "MEAN": self.cf / self.sess_count,
            "TOTAL_OPEX": self.cf,
            "TOTAL_CAPEX": self.embodied_cf,
            "IDLE_CF": self.idle_cf,
            "DEPRECIATION_CF": self.depreciation_cf / self.sess_count,
            "READ_CF": self.read_cf / self.sess_count,
            "WRITE_CF": self.write_cf / self.sess_count,
            "ROUTERS_CF": self.routers_energy_cf / self.sess_count,
            "LINKS_CF": self.links_energy_cf / self.sess_count,
            "EMBODIED_CF": self.embodied_cf / self.sess_count,
            })
        return results

@register_data_collector("CACHE_HIT_RATIO")
class CacheHitRatioCollector(DataCollector):
    """Collector measuring the cache hit ratio, i.e. the portion of content
    requests served by a cache.
    """

    def __init__(self, view, off_path_hits=False, per_node=False, content_hits=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The NetworkView instance
        off_path_hits : bool, optional
            If *True* also records cache hits from caches not on located on the
            shortest path. This metric may be relevant only for some strategies
        content_hits : bool, optional
            If *True* also records cache hits per content instead of just
            globally
        """
        self.view = view
        self.off_path_hits = off_path_hits
        self.per_node = per_node
        self.cont_hits = content_hits
        self.sess_count = 0
        self.cache_hits = 0
        self.serv_hits = 0
        if off_path_hits:
            self.off_path_hit_count = 0
        if per_node:
            self.per_node_cache_hits = collections.defaultdict(int)
            self.per_node_server_hits = collections.defaultdict(int)
        if content_hits:
            self.curr_cont = None
            self.cont_cache_hits = collections.defaultdict(int)
            self.cont_serv_hits = collections.defaultdict(int)

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        self.sess_count += 1
        if self.off_path_hits:
            source = self.view.content_source(content)
            self.curr_path = self.view.shortest_path(receiver, source)
        if self.cont_hits:
            self.curr_cont = content

    @inheritdoc(DataCollector)
    def cache_hit(self, node, **kwargs):
        self.cache_hits += 1
        if self.off_path_hits and node not in self.curr_path:
            self.off_path_hit_count += 1
        if self.cont_hits:
            self.cont_cache_hits[self.curr_cont] += 1
        if self.per_node:
            self.per_node_cache_hits[node] += 1

    @inheritdoc(DataCollector)
    def server_hit(self, node, **kwargs):
        self.serv_hits += 1
        if self.cont_hits:
            self.cont_serv_hits[self.curr_cont] += 1
        if self.per_node:
            self.per_node_server_hits[node] += 1

    @inheritdoc(DataCollector)
    def results(self):
        n_sess = self.cache_hits + self.serv_hits
        hit_ratio = self.cache_hits / n_sess
        results = Tree(**{"MEAN": hit_ratio})
        chrcp["chr"] = hit_ratio
        if self.off_path_hits:
            results["MEAN_OFF_PATH"] = self.off_path_hit_count / n_sess
            results["MEAN_ON_PATH"] = results["MEAN"] - results["MEAN_OFF_PATH"]
        if self.cont_hits:
            cont_set = set(
                list(self.cont_cache_hits.keys()) + list(self.cont_serv_hits.keys())
            )
            cont_hits = {
                i: (
                    self.cont_cache_hits[i]
                    / (self.cont_cache_hits[i] + self.cont_serv_hits[i])
                )
                for i in cont_set
            }
            results["PER_CONTENT"] = cont_hits
        if self.per_node:
            for v in self.per_node_cache_hits:
                self.per_node_cache_hits[v] /= n_sess
            for v in self.per_node_server_hits:
                self.per_node_server_hits[v] /= n_sess
            results["PER_NODE_CACHE_HIT_RATIO"] = self.per_node_cache_hits
            results["PER_NODE_SERVER_HIT_RATIO"] = self.per_node_server_hits
        return results

@register_data_collector("PATH_STRETCH")
class PathStretchCollector(DataCollector):
    """Collector measuring the path stretch, i.e. the ratio between the actual
    path length and the shortest path length.
    """

    def __init__(self, view, cdf=False):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        cdf : bool, optional
            If *True*, also collects a cdf of the path stretch
        """
        self.view = view
        self.cdf = cdf
        self.req_path_len = collections.defaultdict(int)
        self.cont_path_len = collections.defaultdict(int)
        self.sess_count = 0
        self.mean_req_stretch = 0.0
        self.mean_cont_stretch = 0.0
        self.mean_stretch = 0.0
        if self.cdf:
            self.req_stretch_data = collections.deque()
            self.cont_stretch_data = collections.deque()
            self.stretch_data = collections.deque()

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        self.receiver = receiver
        self.source = self.view.content_source(content)
        self.req_path_len = 0
        self.cont_path_len = 0
        self.sess_count += 1

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, **kwargs):
        self.req_path_len += 1

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, **kwargs):
        self.cont_path_len += 1

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        if not success:
            return
        req_sp_len = len(self.view.shortest_path(self.receiver, self.source))
        cont_sp_len = len(self.view.shortest_path(self.source, self.receiver))
        req_stretch = self.req_path_len / req_sp_len
        cont_stretch = self.cont_path_len / cont_sp_len
        stretch = (self.req_path_len + self.cont_path_len) / (req_sp_len + cont_sp_len)
        self.mean_req_stretch += req_stretch
        self.mean_cont_stretch += cont_stretch
        self.mean_stretch += stretch
        if self.cdf:
            self.req_stretch_data.append(req_stretch)
            self.cont_stretch_data.append(cont_stretch)
            self.stretch_data.append(stretch)

    @inheritdoc(DataCollector)
    def results(self):
        results = Tree(
            {
                "MEAN": self.mean_stretch / self.sess_count,
                "MEAN_REQUEST": self.mean_req_stretch / self.sess_count,
                "MEAN_CONTENT": self.mean_cont_stretch / self.sess_count,
            }
        )
        if self.cdf:
            results["CDF"] = cdf(self.stretch_data)
            results["CDF_REQUEST"] = cdf(self.req_stretch_data)
            results["CDF_CONTENT"] = cdf(self.cont_stretch_data)
        return results

@register_data_collector("ESTIMATED_COSTS")
class EstimatedCostsCollector:
    """Object collecting notifications about simulation events and measuring
    relevant metrics.
    """

    def __init__(self, view, **params):
        self.view = view
        self.sess_count = 0
        self.estimated_sess_cost_gain = 0.0
        self.estimated_sess_cost_loss = 0.0
        self.estimated_sess_dep = 0.0
        self.estimated_sess_stor = 0.0
        self.estimated_sess_band = 0.0
        self.estimated_sess_trans = 0.0
        self.estimated_sess_pen = 0.0
        self.estimated_sess_min_gain = 0.0

        self.estimated_cost_gain = 0.0
        self.estimated_cost_loss = 0.0
        self.estimated_dep = 0.0
        self.estimated_stor = 0.0
        self.estimated_band = 0.0
        self.estimated_trans = 0.0
        self.estimated_pen = 0.0
        self.estimated_min_gain = 0.0
        # Define a path to store the cost data
        self.log_file_path = 'estimated_path_log.csv'

    def start_session(self, timestamp, receiver, content, priority):
        self.receiver = receiver
        self.content = content
        self.priority = priority
        self.estimated_sess_cost_gain = 0.0
        self.estimated_sess_cost_loss = 0.0
        self.estimated_sess_dep = 0.0
        self.estimated_sess_stor = 0.0
        self.estimated_sess_band = 0.0
        self.estimated_sess_trans = 0.0
        self.estimated_sess_pen = 0.0
        self.estimated_sess_min_gain = 0.0
        self.sess_count += 1

    def storage_div(self, path, storage_gain, storage_loss, dep, stor, band, trans, pen, min_gain):
        self.estimated_sess_cost_gain += storage_gain
        self.estimated_sess_cost_loss += storage_loss
        self.estimated_sess_dep += dep
        self.estimated_sess_stor += stor
        self.estimated_sess_band += band
        self.estimated_sess_trans += trans
        self.estimated_sess_pen += pen
        self.estimated_sess_min_gain += min_gain
        # with open(self.log_file_path, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow('/')
        #     writer.writerow([path])
        
    def end_session(self, success=True):
        if not success:
            logger.info(f"end failed session")
            return
        self.estimated_cost_gain += self.estimated_sess_cost_gain
        self.estimated_cost_loss += self.estimated_sess_cost_loss
        self.estimated_dep += self.estimated_sess_dep
        self.estimated_stor += self.estimated_sess_stor
        self.estimated_band += self.estimated_sess_band
        self.estimated_trans += self.estimated_sess_trans
        self.estimated_pen += self.estimated_sess_pen
        self.estimated_min_gain +=  self.estimated_sess_min_gain
        # print(f"ESTIMATED bandwidth:{self.estimated_sess_band}")
         # Log the session costs to a CSV file
        # with open(self.log_file_path, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([self.sess_count, self.receiver, self.content, self.estimated_sess_band,
        #                      self.estimated_sess_trans, self.estimated_sess_pen,
        #                      self.estimated_sess_dep, self.estimated_sess_stor])
        logger.info(f"ESTIMATED {self.sess_count}, {self.receiver}, {self.content}, {self.estimated_sess_band}, {self.estimated_sess_trans}, {self.estimated_sess_pen}, {self.estimated_sess_dep}, {self.estimated_sess_stor}")
        

    def results(self):
        results = Tree(
            {
            "STORAGE_GAIN": self.estimated_cost_gain,
            "STORAGE_LOSS": self.estimated_cost_loss,
            "DEPRECIATION":  self.estimated_dep,
            "BANDWIDTH":self.estimated_band,
            "STORAGE": self.estimated_stor,
            "TRANSMISSION":self.estimated_trans,
            "PENALTY": self.estimated_pen,
            "MIN_GAIN": self.estimated_min_gain,
            })
        return results

@register_data_collector("REPLICA_MONITOR")
class LiveReplicaMonitor(DataCollector):
    def __init__(self, view, **params):
        super().__init__(view, **params)
        self.replicas = collections.defaultdict(dict)  # session_id -> {content: count}

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        self.content = content

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        if not success:
            logger.info(f"end failed session")
            return
        """Record replica counts when a session ends"""
        for v in self.view.content_locations(self.content):
            self.replicas[self.content] = len(
                self.view.content_locations(self.content)
            )
    
    def results(self): 
        results = Tree(
            {
            "REPLICAS": self.replicas,
            })
        return results

@register_data_collector("DUMMY")
class DummyCollector(DataCollector):
    """Dummy collector to be used for test cases only."""

    def __init__(self, view):
        """Constructor

        Parameters
        ----------
        view : NetworkView
            The network view instance
        output : stream
            Stream on which debug collector writes
        """
        self.view = view

    @inheritdoc(DataCollector)
    def start_session(self, timestamp, receiver, content, priority):
        self.session = dict(
            timestamp=timestamp,
            receiver=receiver,
            content=content,
            cache_misses=[],
            request_hops=[],
            content_hops=[],
        )

    @inheritdoc(DataCollector)
    def cache_hit(self, node, **kwargs):
        self.session["serving_node"] = node

    @inheritdoc(DataCollector)
    def cache_miss(self, node):
        self.session["cache_misses"].append(node)

    @inheritdoc(DataCollector)
    def server_hit(self, node, **kwargs):
        self.session["serving_node"] = node

    @inheritdoc(DataCollector)
    def request_hop(self, u, v, **kwargs):
        self.session["request_hops"].append((u, v))

    @inheritdoc(DataCollector)
    def content_hop(self, u, v, **kwargs):
        self.session["content_hops"].append((u, v))

    @inheritdoc(DataCollector)
    def end_session(self, success=True):
        self.session["success"] = success

    def session_summary(self):
        """Return a summary of latest session

        Returns
        -------
        session : dict
            Summary of session
        """
        return self.session
