import argparse
import json
import numpy as np

#               per   film  role  film.cvt cty gender
ENTITY_TYPES = ["t0", "t1", "t2", "cvt", "t3", "t4"]

REL_TYPES = ["r0",  # acted_in
             "r1",  # film_name
             "r2",  # char_name
             "r3",  # lives_in
             "r4"   # gender
             ]

ALLOWED_TYPES_MAP = {"t0": [("r0", "cvt"), ("r3", "t3"), ("r4", "t4")],
                     "t1": [("r1", "cvt")],
                     "t3": [("r2", "cvt")]
                     }


class Graph:
    def __init__(self, max_size):
        self.ent2id = {}
        self.entities = []
        self._size = 0
        self.type2ent = {}
        self.ent_types = []
        self.adj_map = {}
        self.adj_map_inv = {}
        self.max_size = max_size

    def size(self):
        return self._size

    def add_edge(self, e1_id, r, e2):
        # if e2 not in self.ent2id:
        #     self.entities.append(e2)
        #     self.ent2id[e2] = self._size
        #     self.ent_types.append(e2_type)
        #     self._size += 1
        #     if e2_type not in self.type2ent:
        #         self.type2ent[e2_type] = [e2]
        #     else:
        #         self.type2ent[e2_type].append(e2)

        if (self.entities[e1_id], r) not in self.adj_map:
            self.adj_map[(self.entities[e1_id], r)] = {e2}
        else:
            self.adj_map[(self.entities[e1_id], r)].add(e2)

        if (e2, r+"_inv") not in self.adj_map_inv:
            self.adj_map_inv[(e2, r+"_inv")] = {self.entities[e1_id]}
        else:
            self.adj_map_inv[(e2, r+"_inv")].add(self.entities[e1_id])

    def add_entity(self, e, e_type):
        assert e not in self.ent2id
        self.entities.append(e)
        self.ent2id[e] = self._size
        self.ent_types.append(e_type)
        self._size += 1
        if e_type not in self.type2ent:
            self.type2ent[e_type] = [e]
        else:
            self.type2ent[e_type].append(e)

    def prune_disconnected_entities(self):
        unused = np.ones(self._size, dtype=np.bool)
        for (e1, r), e2_list in self.adj_map.items():
            unused[self.ent2id[e1]] = 0
            for e2 in e2_list:
                unused[self.ent2id[e2]] = 0
        idx2ent = {v_: k_ for (k_, v_) in self.ent2id.items()}
        st = 0
        entities_removed = []
        entities, ent_types = [], []
        for eid in np.where(unused == 1)[0]:
            e_name = idx2ent[eid]
            e_type = self.ent_types[eid]
            entities.extend(self.entities[st:eid])
            ent_types.extend(self.ent_types[st:eid])
            self.type2ent[e_type].remove(e_name)
            st = eid + 1
            entities_removed.append(e_name)
        # Add back remaining entities and types
        entities.extend(self.entities[st:])
        ent_types.extend(self.ent_types[st:])
        self.entities = entities
        self.ent_types = ent_types
        self.ent2id = {e: ctr for ctr, e in enumerate(self.entities)}
        self._size -= len(entities_removed)
        assert self._size == len(self.entities) == len(self.ent2id)
        print(f"pruned entities ({len(entities_removed)}): {entities_removed}")

    def seed_entities(self, entity_types, prefix=""):
        nodes_per_type = self.max_size // len(entity_types)
        for t_ in entity_types:
            node_id_offset = self._size
            for nid in range(node_id_offset, node_id_offset + nodes_per_type):
                self.add_entity(f"{prefix}e{nid}", t_)


def sample_graph(rng: np.random._generator.Generator, kb_system: dict, current_graph: Graph, src_id, max_depth, density,
                 expanded_arr):
    if max_depth == 0 or current_graph.ent_types[src_id] not in kb_system['allowed_types_map'] or expanded_arr[src_id]:
        return

    expanded_arr[src_id] = 1
    for new_edge_type in kb_system['allowed_types_map'][current_graph.ent_types[src_id]]:
        for tail_ent in current_graph.type2ent.get(new_edge_type[1], []):
            if rng.random() < density:
                current_graph.add_edge(src_id, new_edge_type[0], tail_ent)
                sample_graph(rng, kb_system, current_graph, current_graph.ent2id[tail_ent], max_depth - 1, density,
                             expanded_arr)

    # while rng.random() < density and outgoing_edges_added < 5:
    #     new_edge_type = rng.choice(ALLOWED_TYPES_MAP[current_graph.ent_types[src_id]])
    #     if current_graph.size() < max_size:
    #         tail_ent = rng.choice(current_graph.type2ent.get(new_edge_type[1], []) + [f"e{current_graph.size()}"])
    #     else:
    #         tail_ent = rng.choice(current_graph.type2ent.get(new_edge_type[1], ["retry"]))
    #     if tail_ent == "retry":
    #         continue
    #     current_graph.add_edge(src_id, new_edge_type[0], tail_ent, new_edge_type[1])
    #     sample_graph(rng, current_graph, current_graph.ent2id[tail_ent], max_depth - 1, density, max_size)
    #     outgoing_edges_added += 1


def insert_pattern_in_graph(graph: Graph, kb_system: dict, pattern, rng):
    def _is_bound(n: str, _bindings):
        return not n.startswith('?') or n in _bindings

    def _is_var(n: str):
        return n.startswith('?')

    bindings = {}
    constraints = {}
    edges_bound = np.zeros(len(pattern), dtype=np.bool)
    for e_ctr, (p_1, p_r, p_2) in enumerate(pattern):
        if not _is_var(p_1):
            bindings[p_1] = [p_1]
        else:
            if p_1 not in constraints:
                constraints[p_1] = []
            constraints[p_1].append((e_ctr, 0))
        if not _is_var(p_2):
            bindings[p_2] = [p_2]
        else:
            if p_2 not in constraints:
                constraints[p_2] = []
            constraints[p_2].append((e_ctr, 2))
        if _is_bound(p_1, bindings) and _is_bound(p_2, bindings):
            if (p_1, p_r) not in collected_graph.adj_map:
                collected_graph.adj_map[(p_1, p_r)] = set()
            if (p_2, p_r+"_inv") not in collected_graph.adj_map_inv:
                collected_graph.adj_map_inv[(p_2, p_r+"_inv")] = set()
            if p_2 not in collected_graph.adj_map[(p_1, p_r)]:
                collected_graph.adj_map[(p_1, p_r)].add(p_2)
                collected_graph.adj_map_inv[(p_2, p_r+"_inv")].add(p_1)
            print(f"new edge added: ({p_1}, {p_r}, {p_2})")
            edges_bound[e_ctr] = 1

    while np.sum(edges_bound) < len(pattern):
        for e_ctr, (p_1, p_r, p_2) in enumerate(pattern):
            if edges_bound[e_ctr]:
                continue

            if _is_bound(p_1, bindings) and _is_bound(p_2, bindings):
                new_bindings_p_1, new_bindings_p_2 = [], []
                for e1_bind in bindings[p_1]:
                    if (e1_bind, p_r) not in graph.adj_map:
                        continue
                    valid_e1_bind = False
                    for e2_bind in bindings[p_2]:
                        if e2_bind in graph.adj_map[(e1_bind, p_r)]:
                            valid_e1_bind = True
                            new_bindings_p_2.append(e2_bind)
                    if valid_e1_bind:
                        new_bindings_p_1.append(e1_bind)
                new_bindings_p_1 = sorted(set(new_bindings_p_1))
                new_bindings_p_2 = sorted(set(new_bindings_p_2))
                if len(new_bindings_p_1) == 0 or len(new_bindings_p_2) == 0:
                    # no valid binding
                    print(f"new edge added: ({bindings[p_1][0]}, {p_r}, {bindings[p_2][0]})")
                    graph.add_edge(graph.ent2id[bindings[p_1][0]], p_r, bindings[p_2][0])
                    new_bindings_p_1, new_bindings_p_2 = [bindings[p_1][0]], [bindings[p_2][0]]
                if set(new_bindings_p_1) != set(bindings[p_1]):
                    print(f"binding for {p_1} pruned from {sorted(bindings[p_1])} to {new_bindings_p_1}")
                    bindings[p_1] = new_bindings_p_1
                    if p_1 in constraints:
                        for c_edge in constraints[p_1]:
                            edges_bound[c_edge[0]] = 0

                if set(new_bindings_p_2) != set(bindings[p_2]):
                    print(f"binding for {p_2} pruned from {sorted(bindings[p_2])} to {new_bindings_p_2}")
                    bindings[p_2] = new_bindings_p_2
                    if p_2 in constraints:
                        for c_edge in constraints[p_2]:
                            edges_bound[c_edge[0]] = 0

                edges_bound[e_ctr] = 1
                continue

            if _is_bound(p_1, bindings) and not _is_bound(p_2, bindings):
                new_bindings_p_1 = []
                bindings[p_2] = []
                for e_bind in bindings[p_1]:
                    if (e_bind, p_r) not in graph.adj_map:
                        continue
                    new_bindings_p_1.append(e_bind)
                    bindings[p_2].extend(list(graph.adj_map[(e_bind, p_r)]))
                bindings[p_2] = list(set(bindings[p_2]))

                if not bindings[p_2]:
                    e_bind = bindings[p_1][0]
                    new_bindings_p_1 = [e_bind]
                    tail_ent_type = ''
                    for edge_type in kb_system['allowed_types_map'][graph.ent_types[graph.ent2id[e_bind]]]:
                        if edge_type[0] == p_r:
                            tail_ent_type = edge_type[1]
                            break
                    assert tail_ent_type is not ''
                    tail_ent_options = graph.type2ent[tail_ent_type]
                    tail_ent = tail_ent_options[rng.integers(len(tail_ent_options))]
                    graph.add_edge(graph.ent2id[e_bind], p_r, tail_ent)
                    bindings[p_2] = [tail_ent]
                    print(f"new edge added: ({e_bind}, {p_r}, {tail_ent})")

                if set(new_bindings_p_1) != set(bindings[p_1]):
                    print(f"binding for {p_1} pruned from {sorted(bindings[p_1])} to {new_bindings_p_1}")
                    bindings[p_1] = new_bindings_p_1
                    if p_1 in constraints:
                        for c_edge in constraints[p_1]:
                            edges_bound[c_edge[0]] = 0

                edges_bound[e_ctr] = 1
                continue

            if _is_bound(p_2, bindings) and not _is_bound(p_1, bindings):
                raise NotImplementedError

    print('Verifying pattern bindings...')
    edges_bound_v2 = np.zeros(len(pattern), dtype=np.bool)
    for e_ctr, (p_1, p_r, p_2) in enumerate(pattern):
        if not _is_var(p_1) and not _is_var(p_2):
            if p_2 not in graph.adj_map[(p_1, p_r)]:
                print(f"0: no valid binding for {p_2} in edge {e_ctr}")
                continue
        elif not _is_var(p_1) and _is_var(p_2):
            new_bindings = []
            for e_bind in bindings[p_2]:
                if e_bind in graph.adj_map[(p_1, p_r)]:
                    new_bindings.append(e_bind)
            if len(new_bindings) == 0:
                print(f"1: no valid binding for {p_2} in edge {e_ctr}")
                continue
            if set(new_bindings) != set(bindings[p_2]):
                print(f"1: binding for {p_2} pruned from {sorted(bindings[p_2])} to {sorted(new_bindings)}")
                bindings[p_2] = new_bindings
        elif _is_var(p_1) and not _is_var(p_2):
            raise NotImplementedError
        elif _is_var(p_1) and _is_var(p_2):
            new_bindings_p_1, new_bindings_p_2 = [], []
            for e1_bind in bindings[p_1]:
                if (e1_bind, p_r) not in graph.adj_map:
                    continue
                valid_e1_bind = False
                for e2_bind in bindings[p_2]:
                    if e2_bind in graph.adj_map[(e1_bind, p_r)]:
                        valid_e1_bind = True
                        new_bindings_p_2.append(e2_bind)
                if valid_e1_bind:
                    new_bindings_p_1.append(e1_bind)
            new_bindings_p_1 = sorted(set(new_bindings_p_1))
            new_bindings_p_2 = sorted(set(new_bindings_p_2))
            if len(new_bindings_p_1) == 0:
                print(f"2: no valid binding for {p_1} in edge {e_ctr}")
                continue
            if len(new_bindings_p_2) == 0:
                print(f"2: no valid binding for {p_2} in edge {e_ctr}")
                continue
            if set(new_bindings_p_1) != set(bindings[p_1]):
                print(f"2: binding for {p_1} pruned from {sorted(bindings[p_1])} to {new_bindings_p_1}")
                bindings[p_1] = new_bindings_p_1
            if set(new_bindings_p_2) != set(bindings[p_2]):
                print(f"2: binding for {p_2} pruned from {sorted(bindings[p_2])} to {new_bindings_p_2}")
                bindings[p_2] = new_bindings_p_2
        else:
            raise ValueError("You should not be in this block of code. There is an error in the pattern")
        edges_bound_v2[e_ctr] = 1
    assert np.sum(edges_bound_v2) == len(pattern)
    print('Verified pattern bindings.')
    return bindings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_system", type=str, default=None)
    parser.add_argument("--p", type=float, default=0.4)
    parser.add_argument("--include_inv_edges", action='store_true')
    parser.add_argument("--max_size", type=int, default=40)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.kb_system is None:
        kb_system = {"entity_types": ENTITY_TYPES, "rel_types": REL_TYPES, "allowed_types_map": ALLOWED_TYPES_MAP}
    else:
        kb_system = json.load(open(args.kb_system))
    global_rng = np.random.default_rng(args.seed)

    collected_graph = Graph(args.max_size + 1)
    collected_graph.add_entity("e0", "t0")
    collected_graph.seed_entities(kb_system['entity_types'])
    sample_graph(global_rng, kb_system, collected_graph, 0, args.max_depth, args.p,
                 np.zeros(collected_graph.size(), dtype=np.bool))
    for k, v in collected_graph.adj_map.items():
        print(k, v)
    print('\n---\n')
    res = insert_pattern_in_graph(collected_graph, kb_system, [("e0", "r.t0.t5", "?x0"), ("?x0", "r.t5.t1", "?ans")],
                                  global_rng)
    print(res)
    print('\n---\n')
    collected_graph.prune_disconnected_entities()
    print('\n---\n')
    for k, v in collected_graph.adj_map.items():
        print(k, v)
    print('\n\n********\n\n')

    collected_graph = Graph(args.max_size + 1)
    collected_graph.add_entity("e0", "t0")
    collected_graph.seed_entities(kb_system['entity_types'])
    sample_graph(global_rng, kb_system, collected_graph, 0, args.max_depth, args.p,
                 np.zeros(collected_graph.size(), dtype=np.bool))
    for k, v in collected_graph.adj_map.items():
        print(k, v)
    print('\n---\n')
    res = insert_pattern_in_graph(collected_graph, kb_system, [("e0", "r.t0.t5", "?x0"), ("?x0", "r.t5.t1", "?ans")],
                                  global_rng)
    print(res)
    print('\n---\n')
    collected_graph.prune_disconnected_entities()
    print('\n---\n')
    for k, v in collected_graph.adj_map.items():
        print(k, v)

    # import networkx as nx
    # G = nx.DiGraph()
    # for (e1, r), e2_list in collected_graph.adj_map.items():
    #     for e2 in e2_list:
    #         G.add_edge(e1, e2, label=r)
    # pos = nx.spring_layout(G)
    # nx.draw_networkx(G, pos)
    # edge_labels = nx.get_edge_attributes(G, 'label')
    # nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)
