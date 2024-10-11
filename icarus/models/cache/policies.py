"""Cache replacement policies implementations

This module contains the implementations of all the cache replacement policies
provided by Icarus.
"""

import abc
import copy
import logging
import random
from collections import OrderedDict, defaultdict, deque
import time

from icarus.registry import register_cache_policy
from icarus.util import apportionment, inheritdoc

import numpy as np


__all__ = [
    "Deque",
    "LinkedSet",
    "Cache",
    "NullCache",
    "BeladyMinCache",
    "LruCache",
    "SegmentedLruCache",
    "InCacheLfuCache",
    "PerfectLfuCache",
    "FifoCache",
    "ClimbCache",
    "RandEvictionCache",
    "ARCCache",
    "MARCCache",
    "QMARCCache",
    "insert_after_k_hits_cache",
    "rand_insert_cache",
    "keyval_cache",
    "ttl_cache",
]

logger = logging.getLogger("main")

class Deque(object):
    'Fast searchable queue'

    def __init__(self):
        self.od = OrderedDict()

    def append_left(self, k):
        if k in self.od:
            del self.od[k]
        self.od[k] = None

    def append_by_index(self, index, k):
        if k in self.od:
            del self.od[k]
        # convert the ordered dictionary to a list
        items = list(self.od.items())
        # insert a new element at index 1
        items.insert(index, (k, None))
        self.od = OrderedDict(items)
    
    def pop(self):    
        return self.od.popitem(0)[0]

    def remove(self, k):
        del self.od[k]
    
    def get_without_pop(self):
        if not self.od:
            return None  # or some other default value  
        return next(iter(self.od.items()))[0]
    
    def __len__(self):
        return len(self.od)
    
    def __contains__(self, k):
        return k in self.od

    def __iter__(self):
        return reversed(self.od)

    def __repr__(self):
        return 'Deque(%r)' % (list(self),)
    
    def __clear__(self):
        self.od.clear()
    
    def __index__(self, key):
        keys = list(self.od.keys())
        return keys.index(key)


class LinkedSet:
    """A doubly-linked set, i.e., a set whose entries are ordered and stored
    as a doubly-linked list.

    This data structure is designed to efficiently implement a number of cache
    replacement policies such as LRU and derivatives such as Segmented LRU.

    It provides O(1) time complexity for the following operations: searching,
    remove from any position, move to top, move to bottom, insert after or
    before a given item.
    """

    class _Node:
        """Class implementing a node of the linked list"""

        def __init__(self, val, up=None, down=None):
            """Constructor

            Parameters
            ----------
            val : any hashable type
                The value stored by the node
            up : any hashable type, optional
                The node above in the list
            down : any hashable type, optional
                The node below in the list
            """
            self.val = val
            self.up = up
            self.down = down

    def __init__(self, iterable=[]):
        """Constructor

        Parameters
        ----------
        itaerable : iterable type
            An iterable type to inizialize the data structure.
            It must contain only one instance of each element
        """
        self._top = None
        self._bottom = None
        self._map = {}
        if iterable:
            if len(set(iterable)) < len(iterable):
                raise ValueError("The iterable parameter contains repeated " "elements")
            for i in iterable:
                self.append_bottom(i)

    def __len__(self):
        """Return the number of elements in the linked set

        Returns
        -------
        len : int
            The length of the set
        """
        return len(self._map)

    def __iter__(self):
        """Return an iterator over the set

        Returns
        -------
        reversed : iterator
            An iterator over the set
        """
        cur = self._top
        while cur:
            yield cur.val
            cur = cur.down

    def __reversed__(self):
        """Return a reverse iterator over the set

        Returns
        -------
        reversed : iterator
            A reverse iterator over the set
        """
        cur = self._bottom
        while cur:
            yield cur.val
            cur = cur.up

    def __str__(self):
        """Return a string representation of the set

        Returns
        -------
        str : str
            A string representation of the set
        """
        return (
            self.__class__.__name__
            + "(["
            + "".join("%s, " % str(i) for i in self)[:-2]
            + "])"
        )

    def __contains__(self, k):
        """Return whether the set contains a given item

        Parameters
        ----------
        k : any hashable type
            The item to search

        Returns
        -------
        contains : bool
            *True* if the set contains the item, *False* otherwise
        """
        return k in self._map

    @property
    def top(self):
        """Return the item at the top of the set

        Returns
        -------
        top : any hashable type
            The item at the top or *None* if the set is empty
        """
        return self._top.val if self._top is not None else None

    @property
    def bottom(self):
        """Return the item at the bottom of the set

        Returns
        -------
        bottom : any hashable type
            The item at the bottom or *None* if the set is empty
        """
        return self._bottom.val if self._bottom is not None else None

    def pop_top(self):
        """Pop the item at the top of the set

        Returns
        -------
        top : any hashable type
            The item at the top or *None* if the set is empty
        """
        if self._top is None:  # No elements to pop
            return None
        k = self._top.val
        if self._top == self._bottom:  # One single element
            self._bottom = self._top = None
        else:
            self._top.down.up = None
            self._top = self._top.down
        self._map.pop(k)
        return k

    def pop_bottom(self):
        """Pop the item at the bottom of the set

        Returns
        -------
        bottom : any hashable type
            The item at the bottom or *None* if the set is empty
        """
        if self._bottom is None:  # No elements to pop
            return None
        k = self._bottom.val
        if self._bottom == self._top:  # One single element
            self._top = self._bottom = None
        else:
            self._bottom.up.down = None
            self._bottom = self._bottom.up
        self._map.pop(k)
        return k

    def append_top(self, k):
        """Append an item at the top of the set

        Parameters
        ----------
        k : any hashable type
            The item to append
        """
        if k in self._map:
            raise KeyError("The item %s is already in the set" % str(k))
        n = self._Node(val=k, up=None, down=self._top)
        if self._top == self._bottom is None:
            self._bottom = n
        else:
            self._top.up = n
        self._top = n
        self._map[k] = n

    def append_bottom(self, k):
        """Append an item at the bottom of the set

        Parameters
        ----------
        k : any hashable type
            The item to append
        """
        if k in self._map:
            raise KeyError("The item %s is already in the set" % str(k))
        n = self._Node(val=k, up=self._bottom, down=None)
        if self._top == self._bottom is None:
            self._top = n
        else:
            self._bottom.down = n
        self._bottom = n
        self._map[k] = n

    def move_up(self, k):
        """Move a specified item one position up in the set

        Parameters
        ----------
        k : any hashable type
            The item to move up
        """
        if k not in self._map:
            raise KeyError("Item %s not in the set" % str(k))
        n = self._map[k]
        if n.up is None:  # already on top or there is only one element
            return
        if n.down is None:  # bottom but not top: there are at least two elements
            self._bottom = n.up
        else:
            n.down.up = n.up
        n.up.down = n.down
        new_up = n.up.up
        new_down = n.up
        if new_up:
            new_up.down = n
        else:
            self._top = n
        new_down.up = n
        n.up = new_up
        n.down = new_down

    def move_down(self, k):
        """Move a specified item one position down in the set

        Parameters
        ----------
        k : any hashable type
            The item to move down
        """
        if k not in self._map:
            raise KeyError("Item %s not in the set" % str(k))
        n = self._map[k]
        if n.down is None:  # already at the bottom or there is only one element
            return
        if n.up is None:
            self._top = n.down
        else:
            n.up.down = n.down
        n.down.up = n.up
        new_down = n.down.down
        new_up = n.down
        new_up.down = n
        if new_down is not None:
            new_down.up = n
        else:
            self._bottom = n
        n.up = new_up
        n.down = new_down

    def move_to_top(self, k):
        """Move a specified item to the top of the set

        Parameters
        ----------
        k : any hashable type
            The item to move to the top
        """
        if k not in self._map:
            raise KeyError("Item %s not in the set" % str(k))
        n = self._map[k]
        if n.up is None:  # already on top or there is only one element
            return
        if n.down is None:  # at the bottom, there are at least two elements
            self._bottom = n.up
        else:
            n.down.up = n.up
        n.up.down = n.down
        # Move to top
        n.up = None
        n.down = self._top
        self._top.up = n
        self._top = n

    def move_to_bottom(self, k):
        """Move a specified item to the bottom of the set

        Parameters
        ----------
        k : any hashable type
            The item to move to the bottom
        """
        if k not in self._map:
            raise KeyError("Item %s not in the set" % str(k))
        n = self._map[k]
        if n.down is None:  # already at bottom or there is only one element
            return
        if n.up is None:  # at the top, there are at least two elements
            self._top = n.down
        else:
            n.up.down = n.down
        n.down.up = n.up
        # Move to top
        n.down = None
        n.up = self._bottom
        self._bottom.down = n
        self._bottom = n

    def insert_above(self, i, k):
        """Insert an item one position above a given item already in the set

        Parameters
        ----------
        i : any hashable type
            The item of the set above which the new item is inserted
        k : any hashable type
            The item to insert
        """
        if k in self._map:
            raise KeyError("Item %s already in the set" % str(k))
        if i not in self._map:
            raise KeyError("Item %s not in the set" % str(i))
        n = self._map[i]
        if n.up is None:  # Insert on top
            return self.append_top(k)
        # Now I know I am inserting between two actual elements
        m = self._Node(k, up=n.up, down=n)
        n.up.down = m
        n.up = m
        self._map[k] = m

    def insert_below(self, i, k):
        """Insert an item one position below a given item already in the set

        Parameters
        ----------
        i : any hashable type
            The item of the set below which the new item is inserted
        k : any hashable type
            The item to insert
        """
        if k in self._map:
            raise KeyError("Item %s already in the set" % str(k))
        if i not in self._map:
            raise KeyError("Item %s not in the set" % str(i))
        n = self._map[i]
        if n.down is None:  # Insert on top
            return self.append_bottom(k)
        # Now I know I am inserting between two actual elements
        m = self._Node(k, up=n, down=n.down)
        n.down.up = m
        n.down = m
        self._map[k] = m

    def index(self, k):
        """Return index of a given element.

        This operation has a O(n) time complexity, with n being the size of the
        set.

        Parameters
        ----------
        k : any hashable type
            The item whose index is queried

        Returns
        -------
        index : int
            The index of the item
        """
        if k not in self._map:
            raise KeyError("The item %s is not in the set" % str(k))
        index = 0
        curr = self._top
        while curr:
            if curr.val == k:
                return index
            curr = curr.down
            index += 1
        else:
            raise KeyError(
                "It seems that the item %s is not in the set, "
                "but you should never see this message. "
                "There is something wrong with the code. "
                "Debug it or report it to the developers" % str(k)
            )

    def remove(self, k):
        """Remove an item from the set

        Parameters
        ----------
        k : any hashable type
            The item to remove
        """
        if k not in self._map:
            raise KeyError("Item %s not in the set" % str(k))
        n = self._map[k]
        if self._bottom == n:  # I am trying to remove the last node
            self._bottom = n.up
        else:
            n.down.up = n.up
        if self._top == n:  # I am trying to remove the top node
            self._top = n.down
        else:
            n.up.down = n.down
        self._map.pop(k)

    def clear(self):
        """Empty the set"""
        self._top = None
        self._bottom = None
        self._map.clear()


class Cache:
    """Base implementation of a cache object"""

    @abc.abstractmethod
    def __init__(self, maxlen, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        maxlen : int
            The maximum number of items the cache can store
        """
        raise NotImplementedError("This method is not implemented")

    @abc.abstractmethod
    def __len__(self):
        """Return the number of items currently stored in the cache

        Returns
        -------
        len : int
            The number of items currently in the cache
        """
        raise NotImplementedError("This method is not implemented")

    @property
    @abc.abstractmethod
    def maxlen(self):
        """Return the maximum number of items the cache can store

        Return
        ------
        maxlen : int
            The maximum number of items the cache can store
        """
        raise NotImplementedError("This method is not implemented")

    @abc.abstractmethod
    def dump(self):
        """Return a dump of all the elements currently in the cache possibly
        sorted according to the eviction policy.

        Returns
        -------
        cache_dump : list
            The list of all items currently stored in the cache
        """
        raise NotImplementedError("This method is not implemented")

    def do(self, op, k, *args, **kwargs):
        """Utility method that performs a specified operation on a given item.

        This method allows to perform one of the different operations on an
        item:
         * GET: Retrieve an item
         * PUT: Insert an item
         * UPDATE: Update the value associated to an item
         * DELETE: Remove an item

        Parameters
        ----------
        op : string
            The operation to execute: GET | PUT | UPDATE | DELETE
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        res : bool
            Boolean value being *True* if the operation succeeded or *False*
            otherwise.
        """
        res = {
            "GET": self.get,
            "PUT": self.put,
            "UPDATE": self.put,
            "DELETE": self.remove,
        }[op](k, *args, **kwargs)
        return res if res is not None else False

    @abc.abstractmethod
    def has(self, k, *args, **kwargs):
        """Check if an item is in the cache without changing the internal
        state of the caching object.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : bool
            Boolean value being *True* if the requested item is in the cache
            or *False* otherwise
        """
        raise NotImplementedError("This method is not implemented")

    @abc.abstractmethod
    def get(self, k, *args, **kwargs):
        """Retrieves an item from the cache.

        Differently from *has(k)*, calling this method may change the internal
        state of the caching object depending on the specific cache
        implementation.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : bool
            Boolean value being *True* if the requested item is in the cache
            or *False* otherwise
        """
        raise NotImplementedError("This method is not implemented")

    @abc.abstractmethod
    def put(self, k, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache, it will not be inserted
        again but the internal state of the cache object may change.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        raise NotImplementedError("This method is not implemented")

    @abc.abstractmethod
    def remove(self, k, *args, **kwargs):
        """Remove an item from the cache, if present.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        removed : bool
            *True* if the content was in the cache, *False* if it was not.
        """
        raise NotImplementedError("This method is not implemented")

    @abc.abstractmethod
    def clear(self):
        """Empty the cache"""
        raise NotImplementedError("This method is not implemented")


@register_cache_policy("NULL")
class NullCache(Cache):
    """Implementation of a null cache.

    This is a dummy cache which never stores anything. It is functionally
    identical to a cache with max size equal to 0.
    """

    def __init__(self, maxlen=0, *args, **kwargs):
        """
        Constructor

        Parameters
        ----------
        maxlen : int, optional
            The max length of the cache. This parameters is always ignored
        """
        pass

    def __len__(self):
        """Return the number of items currently stored in the cache.

        Since this is a dummy cache implementation, it is always empty

        Returns
        -------
        len : int
            The number of items currently in the cache. It is always 0.
        """
        return 0

    @property
    def maxlen(self):
        """Return the maximum number of items the cache can store.

        Since this is a dummy cache implementation, this value is 0.

        Returns
        -------
        maxlen : int
            The maximum number of items the cache can store. It is always 0
        """
        return 0

    def dump(self):
        """Return a list of all the elements currently in the cache.

        In this case it is always an empty list.

        Returns
        -------
        cache_dump : list
            An empty list.
        """
        return []

    def has(self, k, *args, **kwargs):
        """Check if an item is in the cache without changing the internal
        state of the caching object.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : bool
            Boolean value being *True* if the requested item is in the cache
            or *False* otherwise. It always returns *False*
        """
        return False

    def get(self, k, *args, **kwargs):
        """Retrieves an item from the cache.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : bool
            Boolean value being *True* if the requested item is in the cache
            or *False* otherwise. It always returns False
        """
        return False

    def put(self, k, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        return None

    def remove(self, k, *args, **kwargs):
        """Remove a specified item from the cache.

        If the element is not present in the cache, no action is taken.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        removed : bool
            *True* if the content was in the cache, *False* if it was not. It
            always return *False*
        """
        return False

    @inheritdoc(Cache)
    def clear(self):
        pass


@register_cache_policy("MIN")
class BeladyMinCache(Cache):
    """Belady's MIN cache replacement policy

    The Belady's MIN policy is the provably optimal cache replacement policy
    under arbitrary workload. Each time an item is inserted into a full cache,
    it evicts the item that will be requested next the latest.

    This policy is not implementable in practice because it requires knowledge
    of future requests, however it is very useful as a theoretical performance
    upper bound.
    """

    @inheritdoc(Cache)
    def __init__(self, maxlen, trace, **kwargs):
        """Constructor

        Parameters
        ----------
        maxlen : int
            The maximum number of items the cache can store
        trace : iterable
            Trace of requests that the cache will be subject to
        """
        self._maxlen = int(maxlen)
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._next = defaultdict(deque)
        for i, k in enumerate(trace):
            self._next[k].append(i)
        for k in self._next.values():
            k.append(np.infty)
        self._cache = {}

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def dump(self):
        return set(self._cache.keys())

    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        self._next[k].popleft()
        return k in self._cache

    def put(self, k, *args, **kwargs):
        if len(self) < self.maxlen:
            self._cache[k] = self._next[k]
            return None
        next_cache = max(self._cache, key=lambda k: self._cache[k][0])
        if self._next[k][0] < self._next[next_cache][0]:
            self._cache.pop(next_cache)
            self._cache[k] = self._next[k]
            return next_cache
        else:
            return None

    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k not in self._cache:
            return False
        self._cache.pop(k)
        return True

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()


@register_cache_policy("LRU")
class LruCache(Cache):
    """Least Recently Used (LRU) cache eviction policy.

    According to this policy, When a new item needs to inserted into the cache,
    it evicts the least recently requested one.
    This eviction policy is efficient for line speed operations because both
    search and replacement tasks can be performed in constant time (*O(1)*).

    This policy has been shown to perform well in the presence of temporal
    locality in the request pattern. However, its performance drops under the
    Independent Reference Model (IRM) assumption (i.e. the probability that an
    item is requested is not dependent on previous requests).
    """

    @inheritdoc(Cache)
    def __init__(self, maxlen, **kwargs):
        self._cache = LinkedSet()
        self._maxlen = int(maxlen)
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def dump(self):
        return list(iter(self._cache))

    def position(self, k, *args, **kwargs):
        """Return the current position of an item in the cache. Position *0*
        refers to the head of cache (i.e. most recently used item), while
        position *maxlen - 1* refers to the tail of the cache (i.e. the least
        recently used item).

        This method does not change the internal state of the cache.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        position : int
            The current position of the item in the cache
        """
        if k not in self._cache:
            raise ValueError("The item %s is not in the cache" % str(k))
        return self._cache.index(k)

    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        # search content over the list
        # if it has it push on top, otherwise return false
        if k not in self._cache:
            return False
        self._cache.move_to_top(k)
        return True

    def put(self, k, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache, it will pushed to the
        top of the cache.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        # if content in cache, push it on top, no eviction
        if k in self._cache:
            self._cache.move_to_top(k)
            return None
        # if content not in cache append it on top
        self._cache.append_top(k)
        return self._cache.pop_bottom() if len(self._cache) > self._maxlen else None

    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k not in self._cache:
            return False
        self._cache.remove(k)
        return True

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()

@register_cache_policy("SLRU")
class SegmentedLruCache(Cache):
    """Segmented Least Recently Used (LRU) cache eviction policy.

    This policy divides the cache space into a number of segments of equal
    size each operating according to an LRU policy. When a new item is inserted
    to the cache, it is placed on the top entry of the bottom segment. Each
    subsequent hit promotes the item to the top entry of the segment above.
    When an item is evicted from a segment, it is demoted to the top entry of
    the segment immediately below. An item is evicted from the cache when it is
    evicted from the bottom segment.

    This policy can be viewed as a sort of combination between an LRU and LFU
    replacement policies as it makes eviction decisions based both frequency
    and recency of item reference.
    """

    def __init__(self, maxlen, segments=2, alloc=None, *args, **kwargs):
        """Constructor

        Parameters
        ----------
        maxlen : int
            The maximum number of items the cache can store
        segments : int
            The number of segments
        alloc : list
            List of floats, summing to 1. Indicates the fraction of overall
            caching space to be allocated to each segment.
        """
        self._maxlen = int(maxlen)
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")
        if not isinstance(segments, int) or segments <= 0 or segments > maxlen:
            raise ValueError("segments must be an integer and 0 < segments <= maxlen")
        if alloc:
            if len(alloc) != segments:
                raise ValueError(
                    "alloc must be an iterable with as many entries as segments"
                )
            if np.abs(np.sum(alloc) - 1) > 0.001:
                raise ValueError("All alloc entries must sum up to 1")
        else:
            alloc = [1 / segments for _ in range(segments)]
        self._segment_maxlen = apportionment(maxlen, alloc)
        self._segment = [LinkedSet() for _ in range(segments)]
        # This map is a dictionary mapping each item in the cache with the
        # segment in which it is located. This is not strictly necessary to
        # locate an item as we could have used the map in each segment.
        # This design choice however speeds up processing at the cost of a
        # moderate increase in memory footprint.
        self._cache = {}

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        if k not in self._cache:
            return False
        seg = self._cache[k]
        if seg == 0:
            self._segment[seg].move_to_top(k)
        else:
            self._segment[seg].remove(k)
            self._segment[seg - 1].append_top(k)
            self._cache[k] = seg - 1
            if len(self._segment[seg - 1]) > self._segment_maxlen[seg - 1]:
                demoted = self._segment[seg - 1].pop_bottom()
                self._segment[seg].append_top(demoted)
                self._cache[demoted] = seg
        return True

    def put(self, k, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache, it will pushed to the
        top of the cache.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        # if content in cache, promote it, no eviction
        if k in self._cache:
            seg = self._cache[k]
            if seg == 0:
                self._segment[seg].move_to_top(k)
            else:
                self._segment[seg].remove(k)
                self._segment[seg - 1].append_top(k)
                self._cache[k] = seg - 1
                if len(self._segment[seg - 1]) > self._segment_maxlen[seg - 1]:
                    demoted = self._segment[seg - 1].pop_bottom()
                    self._segment[seg].append_top(demoted)
                    self._cache[demoted] = seg
            return None
        # if content not in cache append on top of probatory segment and
        # possibly evict LRU item
        self._segment[-1].append_top(k)
        self._cache[k] = len(self._segment) - 1
        if len(self._segment[-1]) > self._segment_maxlen[-1]:
            evicted = self._segment[-1].pop_bottom()
            self._cache.pop(evicted)
            return evicted

    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k not in self._cache:
            return False
        seg = self._cache.pop(k)
        self._segment[seg].remove(k)
        return True

    def position(self, k, *args, **kwargs):
        """Return the current position of an item in the cache. Position *0*
        refers to the head of cache (i.e. most recently used item), while
        position *maxlen - 1* refers to the tail of the cache (i.e. the least
        recently used item).

        This method does not change the internal state of the cache.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        position : int
            The current position of the item in the cache
        """
        if k not in self._cache:
            raise ValueError("The item %s is not in the cache" % str(k))
        seg = self._cache[k]
        position = self._segment[seg].index(k)
        return sum(len(self._segment[i]) for i in range(seg)) + position

    @inheritdoc(Cache)
    def dump(self, serialized=True):
        dump = list(list(iter(s)) for s in self._segment)
        return sum(dump, []) if serialized else dump

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()
        for s in self._segment:
            s.clear()


@register_cache_policy("IN_CACHE_LFU")
class InCacheLfuCache(Cache):
    """In-cache Least Frequently Used (LFU) cache implementation

    The LFU replacement policy keeps a counter associated each item. Such
    counters are increased when the associated item is requested. Upon
    insertion of a new item, the cache evicts the one which was requested the
    least times in the past, i.e. the one whose associated value has the
    smallest value.

    This is an implementation of an In-Cache-LFU, i.e. a cache that keeps
    counters for items only as long as they are in cache and resets the
    counter of an item when it is evicted. This is different from a Perfect-LFU
    policy in which a counter is maintained also when the content is evicted.

    In-cache LFU performs better than LRU under IRM demands.
    However, its implementation is computationally expensive since it
    cannot be implemented in such a way that both search and replacement tasks
    can be executed in constant time. This makes it particularly unfit for
    large caches and line speed operations.
    """

    @inheritdoc(Cache)
    def __init__(self, maxlen, *args, **kwargs):
        self._cache = {}
        self.t = 0
        self._maxlen = int(maxlen)
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def dump(self):
        return sorted(self._cache, key=lambda x: self._cache[x], reverse=True)

    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        if self.has(k):
            freq, t = self._cache[k]
            self._cache[k] = freq + 1, t
            return True
        else:
            return False

    @inheritdoc(Cache)
    def put(self, k, *args, **kwargs):
        if not self.has(k):
            self.t += 1
            self._cache[k] = (1, self.t)
            if len(self._cache) > self._maxlen:
                evicted = min(self._cache, key=lambda x: self._cache[x])
                self._cache.pop(evicted)
                return evicted
        return None

    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k in self._cache:
            self._cache.pop(k)
            return True
        else:
            return False

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()


@register_cache_policy("PERFECT_LFU")
class PerfectLfuCache(Cache):
    """Perfect Least Frequently Used (LFU) cache implementation

    The LFU replacement policy keeps a counter associated each item. Such
    counters are increased when the associated item is requested. Upon
    insertion of a new item, the cache evicts the one which was requested the
    least times in the past, i.e. the one whose associated value has the
    smallest value.

    This is an implementation of a Perfect-LFU, i.e. a cache that keeps
    counters for every item, even for those not in the cache.

    In contrast to LRU, Perfect-LFU has been shown to perform optimally under
    IRM demands. However, its implementation is computationally expensive since
    it cannot be implemented in such a way that both search and replacement
    tasks can be executed in constant time. This makes it particularly unfit
    for large caches and line speed operations.
    """

    @inheritdoc(Cache)
    def __init__(self, maxlen, *args, **kwargs):
        # Dict storing counter for all contents, not only those in cache
        self._counter = {}
        # Set storing only items currently in cache
        self._cache = set()
        self.t = 0
        self._maxlen = int(maxlen)
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def dump(self):
        return sorted(self._cache, key=lambda x: self._counter[x], reverse=True)

    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        self.t += 1
        if k in self._counter:
            freq, t = self._counter[k]
            self._counter[k] = freq + 1, t
        else:
            self._counter[k] = 1, self.t
        if self.has(k):
            return True
        else:
            return False

    @inheritdoc(Cache)
    def put(self, k, *args, **kwargs):
        if not self.has(k):
            if k in self._counter:
                freq, t = self._counter[k]
                self._counter[k] = (freq + 1, t)
            else:
                # If I always call a get before a put, this line should never
                # be executed
                self._counter[k] = (1, self.t)
            self._cache.add(k)
            if len(self._cache) > self._maxlen:
                evicted = min(self._cache, key=lambda x: self._counter[x])
                self._cache.remove(evicted)
                return evicted
        return None

    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k in self._cache:
            self._cache.remove(k)
            return True
        else:
            return False

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()
        self._counter.clear()


@register_cache_policy("FIFO")
class FifoCache(Cache):
    """First In First Out (FIFO) cache implementation.

    According to the FIFO policy, when a new item is inserted, the evicted item
    is the first one inserted in the cache. The behavior of this policy differs
    from LRU only when an item already present in the cache is requested.
    In fact, while in LRU this item would be pushed to the top of the cache, in
    FIFO no movement is performed. The FIFO policy has a slightly simpler
    implementation in comparison to the LRU policy but yields worse performance.
    """

    @inheritdoc(Cache)
    def __init__(self, maxlen, *args, **kwargs):
        self._cache = set()
        self._maxlen = int(maxlen)
        self._d = deque()
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def dump(self):
        return list(self._d)

    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache

    def position(self, k, *args, **kwargs):
        """Return the current position of an item in the cache. Position *0*
        refers to the head of cache (i.e. most recently inserted item), while
        position *maxlen - 1* refers to the tail of the cache (i.e. the least
        recently inserted item).

        This method does not change the internal state of the cache.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        position : int
            The current position of the item in the cache
        """
        i = 0
        for c in self._d:
            if c == k:
                return i
            i += 1
        raise ValueError("The item %s is not in the cache" % str(k))

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        return self.has(k)

    @inheritdoc(Cache)
    def put(self, k, *args, **kwargs):
        evicted = None
        if not self.has(k):
            self._cache.add(k)
            self._d.appendleft(k)
        if len(self._cache) > self.maxlen:
            evicted = self._d.pop()
            self._cache.remove(evicted)
        return evicted

    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k in self._cache:
            self._cache.remove(k)
            self._d.remove(k)
            return True
        else:
            return False

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()
        self._d.clear()


@register_cache_policy("CLIMB")
class ClimbCache(Cache):
    """CLIMB cache implementation

    According to this policy, items are organized in a list. When an item in
    the cache is requested it is moved one position up the list; if already
    on top, nothing is done. When a new item is inserted, it replaces the one
    at the bottom of the list.
    """

    @inheritdoc(Cache)
    def __init__(self, maxlen, *args, **kwargs):
        self._cache = LinkedSet()
        self._maxlen = int(maxlen)
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def dump(self):
        return list(iter(self._cache))

    def position(self, k, *args, **kwargs):
        """Return the current position of an item in the cache. Position *0*
        refers to the head of cache, while position *maxlen - 1* refers to the
        tail of the cache.

        This method does not change the internal state of the cache.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        position : int
            The current position of the item in the cache
        """
        if k not in self._cache:
            raise ValueError("The item %s is not in the cache" % str(k))
        return self._cache.index(k)

    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        # search content over the list
        # if it has it move it one position up, otherwise return false
        if k not in self._cache:
            return False
        self._cache.move_up(k)
        return True

    def put(self, k, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache, it will pushed one
        position up.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        # if content in cache, move one position up, no eviction
        if k in self._cache:
            self._cache.move_up(k)
            return None
        # Note: I am not  sure of the implementation in the case of cache not
        # yet full. In this implementation I am inserting an item to the bottom
        # of the cache when the cache is not full, but I am not sure if I
        # should insert the element on top in this case. I could not find any
        # reference to this in literature. Anyway, I think this should not have
        # an impact on steady state performance
        if len(self._cache) == self._maxlen:
            evicted = self._cache.pop_bottom()
        else:
            evicted = None
        self._cache.append_bottom(k)
        return evicted

    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k not in self._cache:
            return False
        self._cache.remove(k)
        return True

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()


@register_cache_policy("RAND")
class RandEvictionCache(Cache):
    """Random eviction cache implementation.

    This class implements a cache replacement policies which randomly select
    an item to evict when the cache is full. It generally yields poor
    performance in terms of cache hits, especially with non-stationary
    workloads but is sometimes used as baseline and for this reason it has been
    implemented here.

    In case of stationary IRM workloads, the RAND eviction policy provably
    achieves the same cache hit ratio of the FIFO replacement policy.
    """

    @inheritdoc(Cache)
    def __init__(self, maxlen, *args, **kwargs):
        self._maxlen = int(maxlen)
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._cache = set()
        self._a = [None for _ in range(self._maxlen)]

    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)

    @property
    def maxlen(self):
        return self._maxlen

    @inheritdoc(Cache)
    def dump(self):
        return list(self._cache)

    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        return self.has(k)

    @inheritdoc(Cache)
    def put(self, k, *args, **kwargs):
        evicted = None
        if not self.has(k):
            if len(self._cache) == self._maxlen:
                evicted_index = random.randint(0, self.maxlen - 1)
                evicted = self._a[evicted_index]
                self._a[evicted_index] = k
                self._cache.remove(evicted)
            else:
                self._a[len(self._cache)] = k
            self._cache.add(k)
        return evicted

    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k not in self._cache:
            return False
        index = self._a.index(k)
        self._a[index] = self._a[len(self._cache) - 1]
        self._a[len(self._cache) - 1] = None
        self._cache.remove(k)
        return True

    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()

@register_cache_policy("ARC")
class ARCCache():
    @inheritdoc(Cache)
    def __init__(self, maxlen, **kwargs):
        self._cache= {}
        self._maxlen = int(maxlen)
        self.p = 0 
        self.t1 = Deque()
        self.t2 = Deque()
        self.b1 = Deque() 
        self.b2 = Deque()
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")
    
    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)
    
    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen
    
    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache
    
    def replace(self, args):
        """
        If (T1 is not empty) and ((T1 lenght exceeds the target p) or (x is in B2 and T1 lenght == p))
            Delete the LRU page in T1 (also remove it from the cache), and move it to MRU position in B1.
        else
            Delete the LRU page in T2 (also remove it from the cache), and move it to MRU position in B2.
        endif
        """

        if self.t1 and ((args in self.b2 and len(self.t1) == self.p) or (len(self.t1) > self.p)):
            old = self.t1.pop()
            self.b1.append_left(old)
        else:
            old = self.t2.pop()
            self.b2.append_left(old)
        
        # self.lock.acquire()
        del self._cache[old]
        # self.lock.release()

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        logger.info("get"+k.__str__())
        # Case I: x is in T1 or T2.
        #  A cache hit has occurred in ARC(c) and DBL(2c)
        #   Move x to MRU position in T2.
        res = False
        if k in self.t1:
            self.t1.remove(k)
            self.t2.append_left(k)
            res = True

        if k in self.t2:
            self.t2.remove(k)
            self.t2.append_left(k)
            res = True

        return res  # Return value not found in cache
        
    def put(self, k, *args, **kwargs):
        logger.info("put"+k.__str__())
        # Case II: x is in B1
        #  A cache miss has occurred in ARC(c)
        #   ADAPTATION
        #   REPLACE(x)
        #   Move x from B1 to the MRU position in T2 (also fetch x to the cache).
        if k in self.b1:
            self.p = min(self._maxlen, self.p + max(len(self.b2) / len(self.b1), 1))
            self.replace(k)
            self.b1.remove(k)
            self.t2.append_left(k)
            self._cache[k] = True
            return

        # Case III: x is in B2
        #  A cache miss has (also) occurred in ARC(c)
        #   ADAPTATION
        #   REPLACE(x, p)
        #   Move x from B2 to the MRU position in T2 (also fetch x to the cache).

        if k in self.b2:
            self.p = max(0, self.p - max(len(self.b1) / len(self.b2), 1))
            self.replace(k)
            self.b2.remove(k)
            self.t2.append_left(k)
            self._cache[k] = True
            return
        
        # Case IV: x is not in (T1 u B1 u T2 u B2)
        #  A cache miss has occurred in ARC(c) and DBL(2c)

        if len(self.t1) + len(self.b1) == self._maxlen:
            # Case A: L1 (T1 u B1) has exactly c pages.

            if len(self.t1) < self._maxlen:
                # Delete LRU page in B1. REPLACE(x, p)
                self.b1.pop()
                self.replace(k)

            else:
                # Here B1 is empty.
                # Delete LRU page in T1 (also remove it from the cache)
                self._cache.pop(self.t1.pop(), None)

        else:
            # Case B: L1 (T1 u B1) has less than c pages.

            total = len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2)
            if total >= self._maxlen:
                # Delete LRU page in B2, if |T1| + |T2| + |B1| + |B2| == 2c
                if total == (2 * self._maxlen):
                    self.b2.pop()

                # REPLACE(x, p)
                self.replace(k)

        # Finally, fetch x to the cache and move it to MRU position in T1
        self.t1.append_left(k)
        self._cache[k] = True
    
    @inheritdoc(Cache)
    def remove(self, k, *args, **kwargs):
        if k not in self._cache:
            return False
        del self._cache[k]
        return True
    
    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()
        self.t1.__clear__()
        self.t2.__clear__()
        self.b1.__clear__()
        self.b2.__clear__()
        self.p = 0

@register_cache_policy("MARC")
class MARCCache(Cache):
    @inheritdoc(Cache)
    def __init__(self, maxlen, **kwargs):
        self._cache= {}
        self._maxlen = int(maxlen)
        self.p = 0
        self.t1 = Deque()
        self.t2 = Deque()
        self.b1 = Deque() 
        self.b2 = Deque()

        self._caches = kwargs["caches"]
        self._n_caches = len(self._caches)
        self._sizes = [int(cache["size_factor"] * self._maxlen) for cache in self._caches]
        self._beta = {}
        self._tier_m_caches = self.initialize_caches()
        
        for i in range(self._n_caches):
            self._beta[i] = self._sizes[i] / self._sizes[0]
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")

    class TierMCache:
        def __init__(self, maxlen):
            self._maxlen = maxlen
            self.p = 0
            self.t1 = Deque()
            self.t2 = Deque()
            
        def put_t1(self, k):
            a, b = (None, None)
            if len(self.t1) + len(self.t2) >= self._maxlen:
                # if len of T1 is higher than P move from T1 dram to T1 disk
                if self.t1 and len(self.t1) >= self.p:
                    old = self.t1.get_without_pop()
                    self.t1.pop()
                    a, b = (old, "t1")
                # move from T2 dram to T2 disk
                elif self.t2:
                    old = self.t2.get_without_pop()
                    self.t2.pop()
                    a, b = (old, "t2")
                else:
                    old = self.t1.get_without_pop()
                    self.t1.pop()
                    a, b = (old, "t1")
           
            self.t1.append_left(k)
            return (a, b)
        
        def put_t2(self, k):
            a, b = (None, None)
            if len(self.t1) + len(self.t2) >= self._maxlen:
                # if len of T1 is higher than P move from T1 dram to T1 disk
                if self.t1 and len(self.t1) >= self.p:
                    old = self.t1.get_without_pop()
                    self.t1.pop()
                    a, b = (old, "t1")
                # move from T2 dram to T2 disk
                elif self.t2:
                    old = self.t2.get_without_pop()
                    self.t2.pop()
                    a, b = (old, "t2")
                else:
                    old = self.t1.get_without_pop()
                    self.t1.pop()
                    a, b = (old, "t1")
            
            self.t2.append_left(k)
            return (a, b)

    def initialize_caches(self):
        # Iterate through caches and initialize TierMCache with a reference to the next cache
        tier_m_caches = {}
        for i in range(self._n_caches):
            tier_m_caches[i] = self.TierMCache(self._sizes[i])
        return tier_m_caches
    
    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)
    
    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen
    
    @inheritdoc(Cache)
    def dump(self):
        return set(self._cache.keys())
    
    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache
    
    def replace(self, args):
        """
        If (T1 is not empty) and ((T1 lenght exceeds the target p) or (x is in B2 and T1 lenght == p))
            Delete the LRU page in T1 (also remove it from the cache), and move it to MRU position in B1.
        else
            Delete the LRU page in T2 (also remove it from the cache), and move it to MRU position in B2.
        endif
        """

        if self.t1 and ((args in self.b2 and len(self.t1) == self.p) or (len(self.t1) > self.p)):
            old = self.t1.get_without_pop()
            self.t1_pop(old)
            self.b1.append_left(old)
        else:
            old = self.t2.get_without_pop()
            self.t2_pop(old)
            self.b2.append_left(old)
            
        del self._cache[old]

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        logger.info("get"+k.__str__())
        # Case I: x is in T1 or T2.
        #  A cache hit has occurred in ARC(c) and DBL(2c)
        #   Move x to MRU position in T2.
        res = False
        if k in self.t1:
            self.t1_remove(k)
            self.t2_append_left(k)
            res = True

        if k in self.t2:
            self.t2_remove(k)
            self.t2_append_left(k)
            res = True

        return res  # Return value not found in cache
        
    def put(self, k, *args, **kwargs):
        logger.info("put"+k.__str__())
        # Case II: x is in B1
        #  A cache miss has occurred in ARC(c)
        #   ADAPTATION
        #   REPLACE(x)
        #   Move x from B1 to the MRU position in T2 (also fetch x to the cache).
        if k in self.b1:
            self.increment_p(len(self.b1), len(self.b2))
            self.replace(k)
            self.b1.remove(k)
            self.t2_append_left(k)
            self._cache[k] = True
            return

        # Case III: x is in B2
        #  A cache miss has (also) occurred in ARC(c)
        #   ADAPTATION
        #   REPLACE(x, p)
        #   Move x from B2 to the MRU position in T2 (also fetch x to the cache).

        if k in self.b2:
            self.decrement_p(len(self.b1), len(self.b2))
            self.replace(k)
            self.b2.remove(k)
            self.t2_append_left(k)
            self._cache[k] = True
            return
        
        # Case IV: x is not in (T1 u B1 u T2 u B2)
        #  A cache miss has occurred in ARC(c) and DBL(2c)
        if len(self.t1) + len(self.b1) == self._maxlen:
            # Case A: L1 (T1 u B1) has exactly c pages.
            if len(self.t1) < self._maxlen:
                # Delete LRU page in B1. REPLACE(x, p)
                self.b1.pop()
                self.replace(k)
            else:
                # Here B1 is empty.
                # Delete LRU page in T1 (also remove it from the cache)
                old = self.t1.get_without_pop()
                self.t1_pop(old)
                del self._cache[old]
        else:
            # Case B: L1 (T1 u B1) has less than c pages.
            total = len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2)
            if total >= self._maxlen:
                # Delete LRU page in B2, if |T1| + |T2| + |B1| + |B2| == 2c
                if total == (2 * self._maxlen):
                    self.b2.pop()

                # REPLACE(x, p)
                self.replace(k)

        # Finally, fetch x to the cache and move it to MRU position in T1
        self.t1_append_left(k)
        self._cache[k] = True
    
    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()
        self.t1.__clear__()
        self.t2.__clear__()
        self.b1.__clear__()
        self.b2.__clear__()
        self.p = 0

    def t1_pop(self, k):
        self.t1.pop()
        for cache in reversed(list(self._tier_m_caches.values())):
            try:
                if cache.t1:
                    cache.t1.remove(k)
                    break
            except Exception as e:
                pass  

    def t2_pop(self, k):
        self.t2.pop()
        for cache in reversed(list(self._tier_m_caches.values())):
            try:
                if cache.t2:
                    cache.t2.remove(k)
                    break
            except Exception as e:
                pass  

    def t1_remove(self, k):
        self.t1.remove(k)
        for cache in self._tier_m_caches.values():
            try:
                if k in cache.t1:
                    cache.t1.remove(k)
                    break
            except Exception as e:
                pass

    def t2_remove(self, k):
        self.t2.remove(k)
        for cache in self._tier_m_caches.values():
            try:
                if k in cache.t2:
                    cache.t2.remove(k)
                    break
            except Exception as e:
                pass

    def t1_append_left(self, k):
        self.t1.append_left(k)
        for i in range(self._n_caches):
            a, b = self._tier_m_caches[i].put_t1(k)
            if (a,b) != (None, None):
                try:
                    if b =="t1":
                        self._tier_m_caches[i + 1].put_t1(a)
                    else:
                        self._tier_m_caches[i + 1].put_t2(a)
                except Exception as e:
                    pass

    def t2_append_left(self, k):
        self.t2.append_left(k)
        for i in range(self._n_caches):
            a, b = self._tier_m_caches[i].put_t2(k)
            if (a,b) != (None, None):
                try:
                    if b =="t1":
                        self._tier_m_caches[i + 1].put_t1(a)
                    else:
                        self._tier_m_caches[i + 1].put_t2(a)
                except Exception as e:
                    pass

    def increment_p(self, len_b1, len_b2):
        self.p = min(self._maxlen, self.p + max((len_b2 / len_b1) * (sum(self._beta.values())),
                                          sum(self._beta.values())))
        for i in range(self._n_caches):
            self._tier_m_caches[i].p = min(self._tier_m_caches[i]._maxlen, self._tier_m_caches[i].p + max((len_b2 / len_b1) * self._beta[i], self._beta[i]))

    def decrement_p(self, len_b1, len_b2):
        self.p = max(0, self.p - max((len_b1 / len_b2) * (sum(self._beta.values())),
                                     sum(self._beta.values())))
        for i in range(self._n_caches):
            self._tier_m_caches[i].p = max(0, self._tier_m_caches[i].p - max((len_b1 / len_b2) * self._beta[i],  self._beta[i]))
 
@register_cache_policy("QMARC")
class QMARCCache(Cache):
    @inheritdoc(Cache)
    def __init__(self, maxlen, **kwargs):
        logger.info(f"Initializing QMARCCache with maxlen: {maxlen} and kwargs: {kwargs}")
        self._caches = kwargs["tiers"]
        self._n_caches = len(self._caches)
        self._maxlen = int(maxlen)
        self._sizes = [cache["size_factor"] * self._maxlen for cache in self._caches]
        self._names = [cache["name"] for cache in self._caches]
        self._tier_m_caches = self.initialize_caches()
        logger.info(f"Cache initialized. Tiers: {self._tier_m_caches}")
        self._cache= {}
        self.p = 0
        self.t1 = Deque()
        self.t2 = Deque()
        self.b1 = Deque() 
        self.b2 = Deque()
        self._alpha = kwargs["alpha"]
        self._beta = {}
        
        for i in range(self._n_caches):
            self._beta[i] = self._sizes[i] / self._sizes[0]
        if self._maxlen <= 0:
            raise ValueError("maxlen must be positive")

    class TierMCache:
        def __init__(self, name, maxlen):
            self.name = name
            self.p = 0
            self.t1 = Deque()
            self.t2 = Deque()
            self._maxlen = maxlen
            self.last_access = 0.0

        def put_t1(self, k, *args):
            a, b = (None, None)
            if len(self.t1) + len(self.t2) >= self._maxlen:
                # if len of T1 is higher than P move from T1 dram to T1 disk
                if self.t1 and len(self.t1) >= self.p:
                    old = self.t1.get_without_pop()
                    self.t1.pop()
                    a, b = (old, "t1")
                # move from T2 dram to T2 disk
                elif self.t2:
                    old = self.t2.get_without_pop()
                    self.t2.pop()
                    a, b = (old, "t2")
                else:
                    old = self.t1.get_without_pop()
                    self.t1.pop()
                    a, b = (old, "t1")
                
            if args:
                self.t1.append_by_index(args[0], k)
            else:
                self.t1.append_left(k)
            self.last_access = time.time()
            return (a,b)
        
        def put_t2(self, k, *args):
            a, b = (None, None)
            if len(self.t1) + len(self.t2) >= self._maxlen:
                # if len of T1 is higher than P move from T1 dram to T1 disk
                if self.t1 and len(self.t1) >= self.p:
                    old = self.t1.get_without_pop()
                    self.t1.pop()
                    a, b = (old, "t1")
                # move from T2 dram to T2 disk
                elif self.t2:
                    old = self.t2.get_without_pop()
                    self.t2.pop()
                    a, b = (old, "t2")
                else:
                    old = self.t1.get_without_pop()
                    self.t1.pop()
                    a, b = (old, "t1")

            if args:
                self.t2.append_by_index(args[0], k)
            else:
                self.t2.append_left(k)
            self.last_access = time.time()
            return (a, b)

    def initialize_caches(self):
        # Iterate through caches and initialize TierMCache with a reference to the next cache
        tier_m_caches = {}
        for i in range(self._n_caches):
            tier_m_caches[i] = self.TierMCache(self._names[i], self._sizes[i])
        return tier_m_caches
    
    @inheritdoc(Cache)
    def __len__(self):
        return len(self._cache)
    
    @property
    @inheritdoc(Cache)
    def maxlen(self):
        return self._maxlen
    
    @inheritdoc(Cache)
    def dump(self):
        return set(self._cache.keys())
    
    @inheritdoc(Cache)
    def has(self, k, *args, **kwargs):
        return k in self._cache
    
    def replace(self, **args):
        logger.info("replace:%s"%args)
        k = args.get("k")
        min_content = args.get("min_content")
        """
        If (T1 is not empty) and ((T1 lenght exceeds the target p) or (x is in B2 and T1 lenght == p))
            Delete the LRU page in T1 (also remove it from the cache), and move it to MRU position in B1.
        else
            Delete the LRU page in T2 (also remove it from the cache), and move it to MRU position in B2.
        endif
        """

        if self.t1 and ((k in self.b2 and len(self.t1) == self.p) or (len(self.t1) > self.p)):
            old = min_content if min_content is not None else self.t1.get_without_pop()
            logger.info("remove %s from t1", old.__str__())
            self.t1_pop(old)
            self.b1.append_left(old)
        else:
            old = min_content if min_content is not None else  self.t2.get_without_pop()
            logger.info("remove %s from t2", old.__str__())
            self.t2_pop(old)
            self.b2.append_left(old)
        
        del self._cache[old]

    @inheritdoc(Cache)
    def get(self, k, *args, **kwargs):
        # Case I: x is in T1 or T2.
        #  A cache hit has occurred in ARC(c) and DBL(2c)
        #   Move x to MRU position in T2.
        res = False
        if args[0] == 'high':
            if k in self.t1:
                logger.info("move %s from t1 to t2", k.__str__())
                self.t1_remove(k)
                self.t2_append_left(k)
                res = True
            else :
                if k in self.t2:
                    logger.info("promote %s in t2", k.__str__())
                    self.t2_remove(k)
                    self.t2_append_left(k)
                    res = True
                
        else:
            if k in self.t1:
                logger.info("move %s from t1 to t2", k.__str__())
                self.t1_remove(k)
                global_pos = round(len(self.t2) * self._alpha)
                self.t2_append_by_index(k, global_pos)
                res = True
            else :
                if k in self.t2:
                    logger.info("promote %s in t2", k.__str__())
                    current_pos = self.t2.__index__(k)
                    new_pos = int(min(self._maxlen - self.p, current_pos + round(len(self.t2) * self._alpha)))
                    self.t2_remove(k)
                    self.t2_append_by_index(k, new_pos)
                    res = True

        return res  # Return value not found in cache
    
    @inheritdoc(Cache)
    def put(self, k, *args, **kwargs):
        logger.info("put"+k.__str__())
        min_content = kwargs.get("min_content") or None
        # Case II: x is in B1
        #  A cache miss has occurred in ARC(c)
        #   ADAPTATION
        #   REPLACE(x)
        #   Move x from B1 to the MRU position in T2 (also fetch x to the cache).
        logger.info(f"Putting key: {k}")
        if k in self._cache:
            logger.info("%s already in cache, updating value and moving to MRU position." + k.__str__())
            self.get(k, *args, **kwargs)
            return

        res = None
        logger.info(f"Cache before: {self._cache}")
        logger.info(f"T1 before: {self.t1}")
        logger.info(f"T2 before: {self.t2}")
        logger.info(f"B1 before: {self.b1}")
        logger.info(f"B2 before: {self.b2}")
        if k in self.b1:
            self.increment_p(len(self.b1), len(self.b2))
            self.replace(k=k, min_content=min_content)
            self.b1.remove(k)
            logger.info("put %s in t2", k.__str__())
            if args[0] == 'high':
                self.t2_append_left(k)
                self._cache[k] = True
            else:
                global_pos = round(len(self.t2) * self._alpha)
                self.t2_append_by_index(k, global_pos)
                self._cache[k] = True
            logger.info(f"Cache after: {self._cache}")
            logger.info(f"T1 after: {self.t1}")
            logger.info(f"T2 after: {self.t2}")
            logger.info(f"B1 after: {self.b1}")
            logger.info(f"B2 after: {self.b2}")
            return res

        # Case III: x is in B2
        #  A cache miss has (also) occurred in ARC(c)
        #   ADAPTATION
        #   REPLACE(x, p)
        #   Move x from B2 to the MRU position in T2 (also fetch x to the cache).

        if k in self.b2:
            self.decrement_p(len(self.b1), len(self.b2))
            self.replace(k=k, min_content=min_content)
            self.b2.remove(k)
            logger.info("put %s in t2", k.__str__())
            if args[0] == 'high':
                self.t2_append_left(k)
                self._cache[k] = True
            else:
                global_pos = round(len(self.t2) * self._alpha)
                self.t2_append_by_index(k, global_pos)
                self._cache[k] = True
            logger.info(f"Cache after: {self._cache}")
            logger.info(f"T1 after: {self.t1}")
            logger.info(f"T2 after: {self.t2}")
            logger.info(f"B1 after: {self.b1}")
            logger.info(f"B2 after: {self.b2}")
            return res
        
        # Case IV: x is not in (T1 u B1 u T2 u B2)
        #  A cache miss has occurred in ARC(c) and DBL(2c)
        if len(self.t1) + len(self.b1) == self._maxlen:
            # Case A: L1 (T1 u B1) has exactly c pages.
            if len(self.t1) < self._maxlen:
                # Delete LRU page in B1. REPLACE(x, p)
                self.b1.pop()
                self.replace(k=k, min_content=min_content)
            else:
                # Here B1 is empty.
                # Delete LRU page in T1 (also remove it from the cache)
                res = min_content if min_content is not None else self.t1.get_without_pop()
                self.t1_pop(res)
                logger.info("remove %s from t1", res.__str__())
                self._cache.pop(res, None)
        else:
            # Case B: L1 (T1 u B1) has less than c pages.
            total = len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2)
            if total >= self._maxlen:
                # Delete LRU page in B2, if |T1| + |T2| + |B1| + |B2| == 2c
                if total == (2 * self._maxlen):
                    self.b2.pop()

                # REPLACE(x, p)
                self.replace(k=k, min_content=min_content)
        
        logger.info("put %s in t1", k.__str__())
        # Finally, fetch x to the cache and move it to MRU position in T1
        if args[0] == 'high':
            self.t1_append_left(k)
            self._cache[k] = True
        else:
            global_pos = round(len(self.t1) * self._alpha)
            self.t1_append_by_index(k, global_pos)
            self._cache[k] = True
        logger.info(f"Cache after: {self._cache}")
        logger.info(f"T1 after: {self.t1}")
        logger.info(f"T2 after: {self.t2}")
        logger.info(f"B1 after: {self.b1}")
        logger.info(f"B2 after: {self.b2}")
        return res
    
    @inheritdoc(Cache)
    def clear(self):
        self._cache.clear()
        self.t1.__clear__()
        self.t2.__clear__()
        self.b1.__clear__()
        self.b2.__clear__()
        self.p = 0

    def t1_pop(self, k):
        self.t1.pop()
        for cache in reversed(list(self._tier_m_caches.values())):
            try:
                if cache.t1:
                    cache.t1.remove(k)
                    break
            except Exception as e:
                pass  

    def t2_pop(self, k):
        self.t2.pop()
        for cache in reversed(list(self._tier_m_caches.values())):
            try:
                if cache.t2:
                    cache.t2.remove(k)
                    break
            except Exception as e:
                pass  

    def t1_remove(self, k):
        if k in self.t1:
            self.t1.remove(k)
        for cache in self._tier_m_caches.values():
            try:
                if k in cache.t1:
                    cache.t1.remove(k)
                    break
            except Exception as e:
                pass

    def t2_remove(self, k):
        if k in self.t2: 
            self.t2.remove(k)
        for cache in self._tier_m_caches.values():
            try:
                if k in cache.t2:
                    cache.t2.remove(k)
                    break
            except Exception as e:
                pass

    def t1_append_left(self, k):
        self.t1.append_left(k)
        for i in range(self._n_caches):
            a, b = self._tier_m_caches[i].put_t1(k)
            if (a,b) != (None, None):
                try:
                    if b =="t1":
                        self._tier_m_caches[i + 1].put_t1(a)
                    else:
                        self._tier_m_caches[i + 1].put_t2(a)
                except Exception as e:
                    pass

    def t2_append_left(self, k):
        self.t2.append_left(k)
        for i in range(self._n_caches):
            a, b = self._tier_m_caches[i].put_t2(k)
            if (a,b) != (None, None):
                try:
                    if b =="t1":
                        self._tier_m_caches[i + 1].put_t1(a)
                    else:
                        self._tier_m_caches[i + 1].put_t2(a)
                except Exception as e:
                    pass

    def t1_append_by_index(self, k, index):
        self.t1.append_by_index(index, k)
        logger.info("new global pos is = %s" % index)
        t1_tier_length = []
        c_max = []
        tier_nb = 0

        for i in range (self._n_caches):
            t1_tier_length.append(len(self._tier_m_caches[i].t1))
            c_max.append(self._tier_m_caches[i]._maxlen)
        logger.info("t1_tier_length= %s" % t1_tier_length)
        logger.info("c_max: %s"%c_max)
        
        for i in range(len(t1_tier_length) - 1, -1, -1):
            if index <= t1_tier_length[i] != 0:
                tier_nb = i
                break
        
        if index == 0 or t1_tier_length[tier_nb] < c_max[tier_nb]:
            new_index = index
        else:
            new_index = index - t1_tier_length[0]
            for i in range(1, len(t1_tier_length)):
                if new_index >= 0:
                    break
                else:
                    new_index += t1_tier_length[i]

        logger.info("write to t1 of %s at index = %s" % (tier_nb, new_index))
        self._tier_m_caches[tier_nb].put_t1(k, new_index)

    def t2_append_by_index(self, k, index):
        self.t2.append_by_index(index, k)
        logger.info("new global pos is = %s" % index)
        t2_tier_length = []
        c_max = []
        tier_nb = 0

        for i in range (self._n_caches):
            t2_tier_length.append(len(self._tier_m_caches[i].t2))
            c_max.append(self._tier_m_caches[i]._maxlen)
        logger.info("t2_tier_length:%s"%t2_tier_length)
        logger.info("c_max:%s"%c_max)
        for i in range(len(t2_tier_length) -1, -1, -1):
            if index <= t2_tier_length[i] != 0:
                tier_nb = i
                break

        if index == 0 or t2_tier_length[tier_nb] < c_max[tier_nb]:
            new_index = index
        else:
            new_index = index - t2_tier_length[0]
            for i in range(1, len(t2_tier_length)):
                if new_index >= 0:
                    break
                else:
                    new_index += t2_tier_length[i]

        logger.info("write to t2 of %s at index = %s" % (tier_nb, new_index))
        self._tier_m_caches[tier_nb].put_t2(k, new_index)
    
    def increment_p(self, len_b1, len_b2):
        self.p = min(self._maxlen, self.p + max((len_b2 / len_b1) * (sum(self._beta.values())),
                                          sum(self._beta.values())))
        for i in range(self._n_caches):
            self._tier_m_caches[i].p = min(self._tier_m_caches[i]._maxlen, self._tier_m_caches[i].p + max((len_b2 / len_b1) * self._beta[i], self._beta[i]))

    def decrement_p(self, len_b1, len_b2):
        self.p = max(0, self.p - max((len_b1 / len_b2) * (sum(self._beta.values())),
                                     sum(self._beta.values())))
        for i in range(self._n_caches):
            self._tier_m_caches[i].p = max(0, self._tier_m_caches[i].p - max((len_b1 / len_b2) * self._beta[i],  self._beta[i]))

    def t1_get_index_tier(self, index):
        t1_tier_length = []
        c_max = []
        tier_nb = 0

        for i in range (self._n_caches):
            t1_tier_length.append(len(self._tier_m_caches[i].t1))
            c_max.append(self._tier_m_caches[i]._maxlen)
        
        for i in range(len(t1_tier_length) - 1, -1, -1):
            if index <= t1_tier_length[i] != 0:
                tier_nb = i
                break
        
        if index == 0 or t1_tier_length[tier_nb] < c_max[tier_nb]:
            new_index = index
        else:
            new_index = index - t1_tier_length[0]
            for i in range(1, len(t1_tier_length)):
                if new_index >= 0:
                    break
                else:
                    new_index += t1_tier_length[i]
        # logger.info("write to t1 of tier %s at index = %s" % (self._tier_m_caches[tier_nb].name, new_index))
        return tier_nb
    
    def t2_get_index_tier(self, index):
        t2_tier_length = []
        c_max = []
        tier_nb = 0

        for i in range (self._n_caches):
            t2_tier_length.append(len(self._tier_m_caches[i].t2))
            c_max.append(self._tier_m_caches[i]._maxlen)
        
        for i in range(len(t2_tier_length) -1, -1, -1):
            if index <= t2_tier_length[i] != 0:
                tier_nb = i
                break

        if index == 0 or t2_tier_length[tier_nb] < c_max[tier_nb]:
            new_index = index
        else:
            new_index = index - t2_tier_length[0]
            for i in range(1, len(t2_tier_length)):
                if new_index >= 0:
                    break
                else:
                    new_index += t2_tier_length[i]
        # logger.info("write to t2 of %s at index = %s" % (self._tier_m_caches[tier_nb].name, new_index))
        return tier_nb

    def get_tier_index(self, k):
        if k in self.t1 or k in self.b1 or k in self.b2:
            global_pos = round(len(self.t2) * self._alpha)
            return self.t2_get_index_tier(global_pos)
        if k in self.t2:
            current_pos = self.t2.__index__(k)
            new_pos = int(min(self._maxlen - self.p, current_pos + round(len(self.t2) * self._alpha)))
            return self.t2_get_index_tier(new_pos)
        else:
            global_pos = round(len(self.t1) * self._alpha)
            return self.t1_get_index_tier(global_pos)

    def get_tiers_last_access(self):
        tiers_last_access = {}
        for i in range(self._n_caches):
            tiers_last_access[i] = self._tier_m_caches[i].last_access
        return tiers_last_access
           
        
def insert_after_k_hits_cache(cache, k=2, memory=None):
    """Return a cache inserting items only after k requests.

    This methods allows to implement a variant of k-LRU and k-RANDOM policies,
    which insert items in the main cache only at the k-th request. However,
    proper k-LRU and k-RANDOM policies, keep a separate queue of fixed size
    for items being hit the same number of times. For example, let's say k=3,
    then there is a fixed size queue storing all items being hit 1 time and
    another queue for items being hit 2 times. In this implementation there is
    a unique FIFO queue keeping all items being hit < k times.
    The size of this queue is equal to the value of memory parameter. If
    memory is None, then this queue is infinite.

    In the most common case of k=2, this difference of implementation does
    not matter.

    Parameters
    ----------
    cache : Cache
        The instance of a cache to be applied random insertion
    k : int, optional
        The number of hits after which the item is inserted
    memory : int, optional
        The size of the metacache just storing the reference to the item and
        the number of hits, without storing the item itself.

    Returns
    -------
    cache : Cache
        The modified cache instance
    """
    if k < 1:
        raise ValueError("k must be positive")
    if k == 1:
        # This is a corner case, as I always insert at first attempt.
        return cache
    hits = {}
    if memory is not None:
        queue = LinkedSet()
    c_put = cache.put

    def put(item, force_insert=False, *args, **kwargs):
        if force_insert:
            if item in hits:
                hits.pop(item)
                if memory is not None:
                    queue.remove(item)
            return c_put(item)
        if item in hits:
            hits[item] += 1
            if hits[item] < k:
                return None
            else:
                # I got hit enough times, inserting in cache
                hits.pop(item)
                if memory is not None:
                    queue.remove(item)
                return c_put(item)
        else:
            hits[item] = 1
            if memory is not None:
                queue.append_top(item)
                if len(queue) > memory:
                    evicted = queue.pop_bottom()
                    hits.pop(evicted)
            return None

    cache.put = put
    cache.put.__doc__ = c_put.__doc__
    cache._metacache_hits = hits
    if memory is not None:
        cache._metacache_queue = queue
    return cache


def rand_insert_cache(cache, p, seed=None):
    """Return a random insertion cache

    It modifies the instance of a cache object such that items are
    inserted randomly instead of deterministically.

    This function modifies the behavior of the *put* method of a given cache
    instance such that it inserts contents randomly with a given probability

    Parameters
    ----------
    cache : Cache
        The instance of a cache to be applied random insertion
    p : float
        the insert probability
    seed : any hashable type, optional
        The seed of the random number generator

    Returns
    -------
    cache : Cache
        The modified cache instance
    """
    if not isinstance(cache, Cache):
        raise TypeError("cache must be an instance of Cache or its subclasses")
    if p < 0 or p > 1:
        raise ValueError("p must be a value between 0 and 1")
    cache = copy.deepcopy(cache)
    random.seed(seed)
    c_put = cache.put

    def put(k, *args, **kwargs):
        if random.random() < p:
            return c_put(k)

    cache.put = put
    cache.put.__doc__ = c_put.__doc__
    return cache


def keyval_cache(cache):
    """It modifies the instance of a cache object such that items are saved
    together with a value instead of just a key.

    This modifies the signature and/or return types of methods *get*, *put* and
    *dump*. The new format is documented in the docstrings of the modified
    methods of the cache instance.

    This function modifies the contract of methods of Cache objects, which is
    admittedly a bad software engineering practice . It may also lead to bugs
    in a key-value cache implementation if the base key-only cache
    implementation from which it derives has methods calling other methods of
    the same instance.

    Parameters
    ----------
    cache : Cache
        The instance of a cache to be changed to a key-value cache

    Returns
    -------
    cache : Cache
        The modified cache instance
    """
    if not isinstance(cache, Cache):
        raise TypeError("cache must be an instance of Cache or its subclasses")
    if len(cache) > 0:
        raise ValueError("the cache must be empty")
    cache = copy.deepcopy(cache)
    cache._val = {}
    k_put = cache.put
    k_get = cache.get
    k_remove = cache.remove
    k_dump = cache.dump
    k_clear = cache.clear

    def put(k, v, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache with the same value, it
        will not be inserted again but the internal state of the cache object
        may change.

        Parameters
        ----------
        k : any hashable type
            The key of item to be inserted
        v : any hashable type
            The value of item to be inserted

        Returns
        -------
        evicted : tuple
            The key, value tuple of the evicted object or *None* if no contents
            were evicted.
        """
        evicted = k_put(k)
        cache._val[k] = v
        if evicted is not None:
            val = cache._val.pop(evicted)
            return evicted, val

    def get(k, *args, **kwargs):
        """Retrieve an item from the cache.

        Differently from *has(k)*, calling this method may change the internal
        state of the caching object depending on the specific cache
        implementation.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : any hashable type
            The value of the requested object or *None* if it is not in the
            cache
        """
        return cache._val[k] if k_get(k) else None

    def remove(k, *args, **kwargs):
        """Remove an item from the cache, if present

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : any hashable type
            The value of the deleted object or *None* if it was not in the
            cache
        """
        return cache._val.pop(k) if k_remove(k) else None

    def dump(*args, **kwargs):
        """Return a dump of all the elements currently in the cache possibly
        sorted according to the eviction policy.

        Returns
        -------
        cache_dump : list of tuples
            The list of items currently stored in the cache represented as
            key, value pairs
        """
        dump = k_dump()
        return [(k, cache._val[k]) for k in dump]

    def clear():
        k_clear()
        cache._val.clear()

    def value(k, *args, **kwargs):
        """Return the value of item k

        Differently from *get(k)*, calling this method does not change the
        internal state of the cache.

        Parameters
        ----------
        k : any hashable type
            The item looked up in the cache

        Returns
        -------
        v : any hashable type
            The value of the requested object or *None* if it is not in the
            cache
        """
        return cache._val[k] if k in cache._val else None

    cache.put = put
    cache.get = get
    cache.remove = remove
    cache.dump = dump
    cache.clear = clear
    cache.clear.__doc__ = k_clear.__doc__
    cache.value = value

    return cache


def ttl_cache(cache, f_time):
    """Return a TTL cache.

    This function takes as a input a cache policy and returns a new policy
    where items, when inserted, are (optionally) labelled with their expiration
    time and are automatically evicted when their validity expires.

    The time validity is verified against the return value of the callable
    argument *f_time*, which is called whenever a purging is executed.

    This implementation can be used with both real time and simulated time.

    Parameters
    ----------
    cache : Cache
        The instance of a cache to be changed to a TTL cache
    f_time : callable
        A function that returns the current time (simulated or real). The
        return type must be a numerical value, e.g. float

    Returns
    -------
    cache : Cache
        The modified cache instance

    Notes
    -----
    The returned TTL cache performs purging operations only when *has*, *get*,
    *put* and *dump* operations are performed. This ensures correctness when
    normal caches are used with common routing and caching strategies.
    However, if other operations like *position* or *len* are executed,
    results may take into account also expired items. In such cases, it is then
    advisable to execute a *purge* first.
    """
    if not isinstance(cache, Cache):
        raise TypeError("cache must be an instance of Cache or its subclasses")
    if len(cache) > 0:
        raise ValueError("the cache must be empty")
    if not hasattr(f_time, "__call__"):
        raise TypeError("f_time must be callable")
    cache = copy.deepcopy(cache)

    cache.f_time = f_time
    cache.expiry = {}

    cache._exp_list = LinkedSet()

    c_put = cache.put
    c_get = cache.get
    c_has = cache.has
    c_remove = cache.remove
    c_dump = cache.dump
    c_clear = cache.clear

    def _purge_till(expiry):
        """Purge all entries expired before a certain time

        Parameters
        ----------
        expiry : float
            Cutoff expiration time
        """
        while (
            cache._exp_list.bottom is not None
            and cache.expiry[cache._exp_list.bottom] < expiry
        ):
            expired = cache._exp_list.pop_bottom()
            cache.expiry.pop(expired)
            c_remove(expired)

    def purge():
        """Purge all expired items"""
        cache._purge_till(cache.f_time())

    def get(k, *args, **kwargs):
        if c_get(k):
            if cache.f_time() < cache.expiry[k]:
                return True
            else:
                remove(k)
        return False

    def put(k, ttl=None, expires=None, *args, **kwargs):
        """Insert an item in the cache if not already inserted.

        If the element is already present in the cache, it will not be inserted
        again but the internal state of the cache object may change.

        Parameters
        ----------
        k : any hashable type
            The item to be inserted
        ttl : float, optional
            The TTL of the item, i.e. its relative expiration time
        expires : float, optional
            The absolute expiration time of the item. It cannot be used in
            conjunction with ttl. If both ttl and expires are None, then the
            inserted content has infinite TTL.

        Returns
        -------
        evicted : any hashable type
            The evicted object or *None* if no contents were evicted.
        """
        now = cache.f_time()
        if ttl is not None:
            if expires is not None:
                raise ValueError(
                    "Both expires and ttl parameters provided. "
                    "Only one can be provided."
                )
            if ttl <= 0:
                # if TTL is not positive, then do not cache the content at all
                return None
            expires = now + ttl
        else:  # case where TTL is None
            if expires is None:
                # If both TTL and expire are None, then TTL is infinite
                expires = np.infty
            elif expires <= now:
                return None
        # Purge expired items only if cache is full for performance reasons
        if len(cache) == cache.maxlen:
            cache._purge_till(now)
        evicted = c_put(k)
        if evicted is not None:
            cache.expiry.pop(evicted)
            cache._exp_list.remove(evicted)
        if k not in cache.expiry or cache.expiry[k] < expires:
            cache.expiry[k] = expires
            if k in cache._exp_list:
                cache._exp_list.remove(k)
            if len(cache._exp_list) == 0:
                cache._exp_list.append_top(k)
            else:
                for i in cache._exp_list:
                    if expires >= cache.expiry[i]:
                        cache._exp_list.insert_above(i, k)
                        break
                else:
                    cache._exp_list.append_bottom(k)
        return evicted

    def has(k, *args, **kwargs):
        return c_has(k) and cache.f_time() <= cache.expiry[k]

    def remove(k, *args, **kwargs):
        c_remove(k)
        cache.expiry.pop(k)
        cache._exp_list.remove(k)

    def dump():
        """Return a dump of all the elements currently in the cache possibly
        sorted according to the eviction policy.

        Return
        ------
        cache_dump : list of tuples
            The list of items currently stored in the cache represented as
            (key, expiration time) pairs
        """
        cache.purge()
        dump = c_dump()
        return [(k, cache.expiry[k]) for k in dump]

    def clear():
        c_clear()
        cache.expiry.clear()
        cache._exp_list.clear()

    cache._purge_till = _purge_till

    cache.get = get
    cache.put = put
    cache.has = has
    cache.remove = remove
    cache.dump = dump
    cache.clear = clear
    cache.purge = purge
    cache.clear.__doc__ = c_clear.__doc__

    return cache


def ttl_keyval_cache():
    pass
