from typing import Dict, List, Optional, Tuple, Any

# Peak vs Off-Peak platform-transfer penalties (minutes). The same value is reported
# as the average headway at the origin, but it is not added to the total commute time.
PEAK_TRANSFER_PENALTY = 3
OFF_PEAK_TRANSFER_PENALTY = 8


# ============================================================================
# 1. STACK (LIFO) - Used to reconstruct the path in forward order
# ============================================================================
class Stack:
    """Explicit LIFO Stack for backtracking parent pointers from Destination to Origin."""

    def __init__(self) -> None:
        self._items: List[Any] = []

    def push(self, item: Any) -> None:
        self._items.append(item)

    def pop(self) -> Optional[Any]:
        if self.is_empty():
            return None
        return self._items.pop()

    def peek(self) -> Optional[Any]:
        if self.is_empty():
            return None
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)

    def snapshot(self) -> List[Any]:
        """Returns a copy of current stack elements (bottom to top) for visualization."""
        return list(self._items)


# ============================================================================
# 2. QUEUE (FIFO) - Pointer-based O(1) Queue used for Breadth-First Search
# ============================================================================
class _QueueNode:
    def __init__(self, station_id: int) -> None:
        self.station_id: int = station_id
        self.next: Optional["_QueueNode"] = None


class Queue:
    """Explicit O(1) FIFO Queue using linked nodes for BFS traversal."""

    def __init__(self) -> None:
        self.front: Optional[_QueueNode] = None
        self.rear: Optional[_QueueNode] = None
        self._size: int = 0

    def enqueue(self, station_id: int) -> None:
        new_node = _QueueNode(station_id)
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        self._size += 1

    def dequeue(self) -> Optional[int]:
        if self.front is None:
            return None
        removed_id = self.front.station_id
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        self._size -= 1
        return removed_id

    def is_empty(self) -> bool:
        return self.front is None

    def size(self) -> int:
        return self._size


# ============================================================================
# 3. MIN-HEAP (Priority Queue) - Used by Dijkstra's Shortest Path Algorithm
# ============================================================================
class HeapEntry:
    def __init__(self, total_time: float, station_id: int, current_line: str) -> None:
        self.total_time: float = float(total_time)
        self.station_id: int = station_id
        self.current_line: str = current_line


class MinHeap:
    """Explicit Binary Min-Heap ordered by cumulative travel time with float precision."""

    def __init__(self) -> None:
        self._data: List[HeapEntry] = []

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def size(self) -> int:
        return len(self._data)

    def push(self, total_time: float, station_id: int, current_line: str) -> None:
        self._data.append(HeapEntry(total_time, station_id, current_line))
        self._heapify_up(len(self._data) - 1)

    def pop(self) -> Optional[HeapEntry]:
        if self.is_empty():
            return None
        root = self._data[0]
        last = self._data.pop()
        if not self.is_empty():
            self._data[0] = last
            self._heapify_down(0)
        return root

    def _heapify_up(self, idx: int) -> None:
        while idx > 0:
            parent_idx = (idx - 1) // 2
            if self._data[parent_idx].total_time <= self._data[idx].total_time:
                break
            self._data[parent_idx], self._data[idx] = self._data[idx], self._data[parent_idx]
            idx = parent_idx

    def _heapify_down(self, idx: int) -> None:
        n = len(self._data)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx

            if left < n and self._data[left].total_time < self._data[smallest].total_time:
                smallest = left
            if right < n and self._data[right].total_time < self._data[smallest].total_time:
                smallest = right

            if smallest == idx:
                break

            self._data[idx], self._data[smallest] = self._data[smallest], self._data[idx]
            idx = smallest


# ============================================================================
# 4. GRAPH (Adjacency List) - Models Stations (Vertices) & Tracks (Edges)
# ============================================================================
class Edge:
    def __init__(self, to_id: int, distance_km: float, base_time_mins: int, line_name: str) -> None:
        self.to_id = to_id
        self.distance_km = distance_km
        self.base_time_mins = base_time_mins
        self.delay_mins = 0
        self.is_closed = False
        self.line_name = line_name


class MetroGraph:
    def __init__(self) -> None:
        self.stations: Dict[int, Dict[str, Any]] = {}
        self.adj_list: Dict[int, List[Edge]] = {}
        self._initialize_ahmedabad_network()

    def add_station(self, st_id: int, name: str, corridor: str, is_ug: bool, x: int, y: int) -> None:
        self.stations[st_id] = {
            "id": st_id,
            "name": name,
            "corridor": corridor,
            "is_underground": is_ug,
            "is_interchange": False,  # Computed dynamically from incident edges
            "x": x,
            "y": y,
        }
        self.adj_list[st_id] = []

    def _update_interchange_status_for(self, st_id: int) -> None:
        if st_id in self.stations:
            distinct_lines = {edge.line_name for edge in self.adj_list.get(st_id, [])}
            self.stations[st_id]["is_interchange"] = len(distinct_lines) > 1

    def _refresh_all_interchanges(self) -> None:
        for st_id in self.stations:
            self._update_interchange_status_for(st_id)

    def add_edge(self, u: int, v: int, dist_km: float, time_mins: int, line_name: str) -> None:
        self.adj_list[u].append(Edge(v, dist_km, time_mins, line_name))
        self.adj_list[v].append(Edge(u, dist_km, time_mins, line_name))
        self._update_interchange_status_for(u)
        self._update_interchange_status_for(v)

    def _initialize_ahmedabad_network(self) -> None:
        """Seeds all 54 stations across the Ahmedabad-Gandhinagar Metro network."""
        station_seeds = [
            # 1. BLUE LINE (East-West Corridor: Vastral Gam [0] to Thaltej Gam [17] - 18 Stations, 21.2 km)
            (0, "Vastral Gam", "Blue Line", False, 1240, 790),
            (1, "Nirant Cross Road", "Blue Line", False, 1180, 750),
            (2, "Vastral", "Blue Line", False, 1120, 710),
            (3, "Rabari Colony", "Blue Line", False, 1060, 670),
            (4, "Amraiwadi", "Blue Line", False, 1000, 635),
            (5, "Apparel Park", "Blue Line", False, 940, 600),
            (6, "Kankaria East", "Blue Line", True, 865, 600),
            (7, "Kalupur Railway Stn", "Blue Line", True, 790, 600),
            (8, "Gheekanta", "Blue Line", True, 715, 600),
            (9, "Shahpur", "Blue Line", True, 645, 600),
            (10, "Old High Court", "Blue Line", False, 570, 600),
            (11, "SP Stadium", "Blue Line", False, 495, 600),
            (12, "Commerce Six Road", "Blue Line", False, 420, 600),
            (13, "Gujarat University", "Blue Line", False, 345, 600),
            (14, "Gurukul Road", "Blue Line", False, 270, 600),
            (15, "Doordarshan Kendra", "Blue Line", False, 195, 600),
            (16, "Thaltej", "Blue Line", False, 120, 600),
            (17, "Thaltej Gam", "Blue Line", False, 50, 600),

            # 2. RED LINE SOUTH (APMC Vasna [18] to Gandhigram [23])
            (18, "APMC Vasna", "Red Line", False, 570, 920),
            (19, "Jivraj Park", "Red Line", False, 570, 868),
            (20, "Rajiv Nagar", "Red Line", False, 570, 816),
            (21, "Shreyas", "Red Line", False, 570, 764),
            (22, "Paldi", "Red Line", False, 570, 712),
            (23, "Gandhigram", "Red Line", False, 570, 656),

            # 3. RED LINE NORTH - AHMEDABAD (Usmanpura [24] to Motera Stadium [31])
            (24, "Usmanpura", "Red Line", False, 570, 544),
            (25, "Vijay Nagar", "Red Line", False, 570, 496),
            (26, "Vadaj", "Red Line", False, 570, 448),
            (27, "Ranip", "Red Line", False, 570, 400),
            (28, "Sabarmati Railway Stn", "Red Line", False, 570, 352),
            (29, "AEC", "Red Line", False, 570, 304),
            (30, "Sabarmati", "Red Line", False, 570, 256),
            (31, "Motera Stadium", "Red Line", False, 570, 210),

            # 4. RED LINE PHASE 2 - GANDHINAGAR (Koteshwar Road [32] to Mahatma Mandir [51])
            (32, "Koteshwar Road", "Red Line", False, 570, 165),
            (33, "Vishwakarma College", "Red Line", False, 570, 120),
            (34, "Tapovan Circle", "Red Line", False, 570, 75),
            (35, "Narmada Canal", "Red Line", False, 635, 75),
            (36, "Koba Circle", "Red Line", False, 700, 75),
            (37, "Juna Koba", "Red Line", False, 765, 75),
            (38, "Koba Gam", "Red Line", False, 830, 75),
            (39, "GNLU", "Red Line", False, 895, 75),
            (40, "Raysan", "Red Line", False, 895, 130),
            (41, "Randesan", "Red Line", False, 895, 180),
            (42, "Dholakuva Circle", "Red Line", False, 895, 230),
            (43, "Infocity", "Red Line", False, 895, 280),
            (44, "Sector-1", "Red Line", False, 895, 330),
            (45, "Sector-10A", "Red Line", False, 975, 330),
            (46, "Sachivalaya", "Red Line", False, 1055, 330),
            (47, "Akshardham", "Red Line", False, 1135, 330),
            (48, "Juna Sachivalaya", "Red Line", False, 1135, 275),
            (49, "Sector-16", "Red Line", False, 1135, 220),
            (50, "Sector-24", "Red Line", False, 1135, 165),
            (51, "Mahatma Mandir", "Red Line", False, 1135, 110),

            # 5. VIOLET / PURPLE LINE (GNLU [39] to GIFT City [53])
            (52, "PDEU", "Violet Line", False, 995, 75),
            (53, "GIFT City", "Violet Line", False, 1095, 75),
        ]

        for st_id, name, corridor, is_ug, x, y in station_seeds:
            self.add_station(st_id, name, corridor, is_ug, x, y)

        # Connect Blue Line (0 -> 17)
        for i in range(17):
            self.add_edge(i, i + 1, 1.25, 3, "Blue Line")

        # Connect Red Line South (18 -> 23 -> 10) & North (10 -> 24 -> 51)
        for i in range(18, 23):
            self.add_edge(i, i + 1, 1.30, 3, "Red Line")
        self.add_edge(23, 10, 1.40, 3, "Red Line")  # Gandhigram <-> Old High Court
        self.add_edge(10, 24, 1.40, 3, "Red Line")  # Old High Court <-> Usmanpura
        for i in range(24, 51):
            self.add_edge(i, i + 1, 1.35, 3, "Red Line")

        # Connect Violet Line Branch (39 -> 52 -> 53)
        self.add_edge(39, 52, 2.30, 4, "Violet Line")
        self.add_edge(52, 53, 2.60, 4, "Violet Line")

        # ========================================================================
        # MULTI-CORRIDOR RING LINKS & CROSS-CITY BYPASSES
        # ========================================================================
        # 1. Western Circular Ring (Paldi <-> Commerce Six Road directly)
        # 1 Hop for BFS, but 16 mins (Dijkstra prefers Metro during Peak)
        self.add_edge(22, 12, 5.20, 16, "Western Ring Link")
        self.add_edge(21, 13, 4.80, 14, "Western Ring Link") # Shreyas <-> Gujarat Univ

        # 2. Central Riverfront Express & Cross Connectors (Orange Line)
        self.add_edge(18, 5, 8.20, 14, "Orange Line")        # APMC Vasna <-> Apparel Park
        self.add_edge(22, 6, 3.50, 6, "Orange Line")         # Paldi <-> Kankaria East
        self.add_edge(22, 7, 4.20, 8, "Orange Line")         # Paldi <-> Kalupur Rly Stn

        # 3. West-North Express (Green Line)
        self.add_edge(17, 29, 6.50, 10, "Green Line")        # Thaltej Gam <-> AEC
        self.add_edge(15, 26, 4.20, 7, "Green Line")         # Doordarshan Kendra <-> Vadaj
        self.add_edge(13, 26, 3.80, 8, "Green Line")         # Gujarat Univ <-> Vadaj

        # 4. Outer Ring & Capital Links (Yellow & Violet Lines)
        self.add_edge(31, 52, 9.00, 15, "Yellow Line")       # Motera Stadium <-> PDEU
        self.add_edge(28, 0, 11.50, 20, "Yellow Line")       # Sabarmati Rly <-> Vastral Gam
        self.add_edge(53, 46, 5.00, 9, "Violet Line")        # GIFT City <-> Sachivalaya

        # 5. Existing Transit Feeder Connectors
        self.add_edge(12, 27, 4.80, 12, "Feeder Connector")  # Commerce Six Rd <-> Ranip
        self.add_edge(7, 28, 5.50, 14, "Feeder Connector")   # Kalupur Rly <-> Sabarmati Rly

        self._refresh_all_interchanges()

    @staticmethod
    def calculate_fares(distance_km: float) -> Dict[str, float]:
        """Distance-slab fare model (Rs. 5 to Rs. 40) with a 10% NCMC discount.

        The slab table is a configurable model, not an official GMRC tariff.
        Distance is rounded to 2 decimals first: summing floats such as 0.1 twenty
        times gives 2.0000000000000004 and would otherwise jump to the next slab.
        """
        distance_km = round(distance_km, 2)
        if distance_km <= 2.0:
            base = 5.0
        elif distance_km <= 5.0:
            base = 10.0
        elif distance_km <= 8.0:
            base = 15.0
        elif distance_km <= 12.0:
            base = 20.0
        elif distance_km <= 16.0:
            base = 25.0
        elif distance_km <= 22.0:
            base = 30.0
        elif distance_km <= 30.0:
            base = 35.0
        else:
            base = 40.0

        return {
            "token_fare": base,
            "ncmc_fare": round(base * 0.90, 2),  # 10% NCMC Smart Card Discount
            "tourist_pass": 100.0,               # Rs. 100 Unlimited Daily Tourist Pass
        }

    def _reconstruct_path_with_stack(
        self,
        forward_states: List[Tuple[int, str]],
        stack_pushed_order: List[str],
        total_dist_km: float,
        total_time_mins: int,
        visited_order: List[int],
        is_peak: bool,
        algorithm_name: str,
    ) -> Dict[str, Any]:
        """
        Uses the popped states from the custom LIFO Stack to build itinerary steps,
        detecting line transfers accurately at the interchange stations.
        """
        forward_ids = [st[0] for st in forward_states]
        n = len(forward_states)

        steps = []
        transfers = 0

        for idx in range(n):
            st_id = forward_states[idx][0]
            is_transfer = False
            prev_line = None

            if idx == 0:
                # Origin station: boards the outgoing line towards the next station
                line_used = forward_states[1][1] if n > 1 else "Origin"
            elif idx == n - 1:
                # Destination station: arrived on the incoming line
                line_used = forward_states[idx][1]
            else:
                # Intermediate station: transfer happens if arrived on line A and departing on line B
                in_line = forward_states[idx][1]
                out_line = forward_states[idx + 1][1]
                if in_line != out_line:
                    is_transfer = True
                    transfers += 1
                    prev_line = in_line
                    line_used = out_line  # Board this line at the transfer station
                else:
                    line_used = out_line

            steps.append({
                "step": idx,
                "station": self.stations[st_id],
                "line_used": line_used,
                "transfer_here": is_transfer,
                "previous_line": prev_line,
            })

        penalty = PEAK_TRANSFER_PENALTY if is_peak else OFF_PEAK_TRANSFER_PENALTY
        transfer_time = transfers * penalty
        ride_time = int(total_time_mins) - transfer_time   # base time + signal delays
        total_time = int(total_time_mins)                  # == the cost the algorithm optimised
        fares = self.calculate_fares(total_dist_km)

        return {
            "found": True,
            "algorithm": algorithm_name,
            "path_ids": forward_ids,
            "steps": steps,
            "visited_order": visited_order,
            "stack_trace": {
                "pushed_lifo_bottom_to_top": stack_pushed_order,
                "popped_forward_order": [self.stations[s]["name"] for s in forward_ids],
            },
            "metrics": {
                "total_stops": max(0, len(forward_ids) - 1),
                "transfers": transfers,
                "distance_km": round(total_dist_km, 2),
                "in_train_time_mins": ride_time,
                "transfer_time_mins": transfer_time,
                "total_commute_mins": total_time,
                # Informational only (average platform headway at origin); NOT part of the total
                "frequency_wait_mins": penalty,
                "total_with_wait_mins": total_time + penalty,
                "operating_hours": "06:20 AM – 10:05 PM",
                **fares,
            },
        }

    # ========================================================================
    # ALGORITHM 1: DIJKSTRA'S SHORTEST PATH + MIN-HEAP PRIORITY QUEUE
    # ========================================================================
    def dijkstra_fastest_route(self, src: int, dest: int, is_peak: bool = True) -> Dict[str, Any]:
        transfer_penalty = PEAK_TRANSFER_PENALTY if is_peak else OFF_PEAK_TRANSFER_PENALTY
        min_time: Dict[Tuple[int, str], float] = {}
        parent: Dict[Tuple[int, str], Tuple[int, str]] = {}
        dist_acc: Dict[Tuple[int, str], float] = {}
        visited_set = set()
        visited_order: List[int] = []

        pq = MinHeap()
        start_state = (src, "")
        pq.push(0.0, src, "")
        min_time[start_state] = 0.0
        dist_acc[start_state] = 0.0

        best_dest_state: Optional[Tuple[int, str]] = None

        while not pq.is_empty():
            curr = pq.pop()
            if curr is None:
                break
            u, cur_time, cur_line = curr.station_id, curr.total_time, curr.current_line

            # Skip outdated heap entries
            if cur_time > min_time.get((u, cur_line), float("inf")):
                continue

            if u not in visited_set:
                visited_set.add(u)
                visited_order.append(u)

            # Early termination: The Min-Heap guarantees the first time dest is popped is optimal!
            if u == dest:
                best_dest_state = (u, cur_line)
                break

            for edge in self.adj_list[u]:
                if edge.is_closed:
                    continue
                v = edge.to_id
                penalty = transfer_penalty if (cur_line and cur_line != edge.line_name) else 0
                new_time = cur_time + edge.base_time_mins + edge.delay_mins + penalty
                next_state = (v, edge.line_name)

                if new_time < min_time.get(next_state, float("inf")):
                    min_time[next_state] = new_time
                    dist_acc[next_state] = dist_acc.get((u, cur_line), 0.0) + edge.distance_km
                    parent[next_state] = (u, cur_line)
                    pq.push(new_time, v, edge.line_name)

        if best_dest_state is None:
            return {
                "found": False,
                "visited_order": visited_order,
                "message": "No active route available between the selected stations due to track closures.",
            }

        # Backtrack using custom LIFO Stack directly with 2D state space (no unsafe 1D flattening)
        path_stack = Stack()
        curr_state: Optional[Tuple[int, str]] = best_dest_state
        while curr_state is not None:
            path_stack.push(curr_state)
            curr_state = parent.get(curr_state)

        # Bottom to top snapshot (Destination at bottom, Origin at top)
        stack_pushed_order = [self.stations[st[0]]["name"] for st in path_stack.snapshot()]

        # Pop to reverse order: LIFO pop yields forward path (Origin to Destination)
        forward_states: List[Tuple[int, str]] = []
        while not path_stack.is_empty():
            st = path_stack.pop()
            if st is not None:
                forward_states.append(st)

        return self._reconstruct_path_with_stack(
            forward_states=forward_states,
            stack_pushed_order=stack_pushed_order,
            total_dist_km=dist_acc[best_dest_state],
            total_time_mins=int(min_time[best_dest_state]),
            visited_order=visited_order,
            is_peak=is_peak,
            algorithm_name="Dijkstra's Algorithm (Optimal Transfer Handling)",
        )

    # ========================================================================
    # ALGORITHM 2: BREADTH-FIRST SEARCH (BFS) + LINKED FIFO QUEUE
    # ========================================================================
    def bfs_minimum_stops(self, src: int, dest: int, is_peak: bool = True) -> Dict[str, Any]:
        """Fewest stops; among all minimum-stop routes, the one with the least time.

        The FIFO queue holds stations (hop levels). Per station we keep one record
        for each incoming line, because the transfer penalty of the NEXT edge depends
        on the line used to arrive. Keeping only one record per station (the old
        approach) can discard the arrival line that avoids a later transfer.
        Only arrivals at the station's minimum hop level are kept: every prefix of a
        minimum-hop path is itself a minimum-hop path, so nothing optimal is lost.
        """
        transfer_penalty = PEAK_TRANSFER_PENALTY if is_peak else OFF_PEAK_TRANSFER_PENALTY
        dist_hops: Dict[int, int] = {src: 0}
        lines_at: Dict[int, List[str]] = {src: [""]}
        time_acc: Dict[Tuple[int, str], int] = {(src, ""): 0}
        dist_acc: Dict[Tuple[int, str], float] = {(src, ""): 0.0}
        parent: Dict[Tuple[int, str], Tuple[int, str]] = {}
        visited_order: List[int] = []

        bfs_queue = Queue()
        bfs_queue.enqueue(src)

        found = False
        while not bfs_queue.is_empty():
            u = bfs_queue.dequeue()
            if u is None:
                break
            visited_order.append(u)

            # All level-(h-1) stations were dequeued earlier (FIFO), so every
            # arrival record of u is final by now.
            if u == dest:
                found = True
                break

            next_hops = dist_hops[u] + 1
            for cur_line in lines_at[u]:
                cur_state = (u, cur_line)
                for edge in self.adj_list[u]:
                    if edge.is_closed:
                        continue
                    v = edge.to_id
                    if v not in dist_hops:
                        dist_hops[v] = next_hops
                        lines_at[v] = []
                        bfs_queue.enqueue(v)
                    if dist_hops[v] != next_hops:
                        continue  # a longer route to v; irrelevant for min-hop paths

                    penalty = transfer_penalty if (cur_line and cur_line != edge.line_name) else 0
                    cand_time = time_acc[cur_state] + edge.base_time_mins + edge.delay_mins + penalty
                    next_state = (v, edge.line_name)

                    if next_state not in time_acc:
                        lines_at[v].append(edge.line_name)
                    if next_state not in time_acc or cand_time < time_acc[next_state]:
                        time_acc[next_state] = cand_time
                        dist_acc[next_state] = dist_acc[cur_state] + edge.distance_km
                        parent[next_state] = cur_state

        if not found:
            return {
                "found": False,
                "visited_order": visited_order,
                "message": "No active route available between the selected stations due to track closures.",
            }

        best_state = min(((dest, ln) for ln in lines_at[dest]), key=lambda s: time_acc[s])

        # Backtrack using custom LIFO Stack over (station, line) states
        path_stack = Stack()
        curr_state: Optional[Tuple[int, str]] = best_state
        while curr_state is not None:
            path_stack.push(curr_state)
            curr_state = parent.get(curr_state)

        stack_pushed_order = [self.stations[st[0]]["name"] for st in path_stack.snapshot()]

        forward_states: List[Tuple[int, str]] = []
        while not path_stack.is_empty():
            st = path_stack.pop()
            if st is not None:
                forward_states.append(st)

        return self._reconstruct_path_with_stack(
            forward_states=forward_states,
            stack_pushed_order=stack_pushed_order,
            total_dist_km=dist_acc[best_state],
            total_time_mins=time_acc[best_state],
            visited_order=visited_order,
            is_peak=is_peak,
            algorithm_name="Breadth-First Search (BFS) + FIFO Queue (Fewest Station Hops)",
        )

    def update_track(self, u: int, v: int, is_closed: bool, delay_mins: int) -> bool:
        updated = False
        for edge in self.adj_list.get(u, []):
            if edge.to_id == v:
                edge.is_closed = is_closed
                edge.delay_mins = max(0, delay_mins)
                updated = True
        for edge in self.adj_list.get(v, []):
            if edge.to_id == u:
                edge.is_closed = is_closed
                edge.delay_mins = max(0, delay_mins)
                updated = True
        return updated

    def reset_all_tracks(self) -> None:
        for edges in self.adj_list.values():
            for edge in edges:
                edge.is_closed = False
                edge.delay_mins = 0

    def serialize_network(self) -> Dict[str, Any]:
        edges_out = []
        for u, edges in self.adj_list.items():
            for e in edges:
                if u < e.to_id:
                    edges_out.append({
                        "u": u,
                        "v": e.to_id,
                        "u_name": self.stations[u]["name"],
                        "v_name": self.stations[e.to_id]["name"],
                        "distance_km": e.distance_km,
                        "base_time_mins": e.base_time_mins,
                        "delay_mins": e.delay_mins,
                        "is_closed": e.is_closed,
                        "line_name": e.line_name,
                    })
        return {
            "stations": list(self.stations.values()),
            "edges": edges_out,
        }
