import argparse
import json
import numpy as np
import os
import shutil

from sample_synthetic_graph import Graph, sample_graph, insert_pattern_in_graph, \
    ENTITY_TYPES, REL_TYPES, ALLOWED_TYPES_MAP

PATTERN_TYPES = ["2p", "3p", "2i", "ip", "pi"]


def ground_pattern(pattern_type, kb_system, rng):
    if pattern_type == '2p':
        src = rng.choice(list(kb_system["allowed_types_map"].keys()))
        r1, t1 = rng.choice(kb_system["allowed_types_map"][src])
        r2, _ = rng.choice([edge_type for edge_type in kb_system["allowed_types_map"][t1] if edge_type[1] != src])
        return [("e0", r1, "?x0"), ("?x0", r2, "?ans")], [("e0", src)]
    elif pattern_type == '3p':
        src = rng.choice(list(kb_system["allowed_types_map"].keys()))
        r1, t1 = rng.choice(kb_system["allowed_types_map"][src])
        r2, t2 = rng.choice([edge_type for edge_type in kb_system["allowed_types_map"][t1] if edge_type[1] != src])
        r3, _ = rng.choice([edge_type for edge_type in kb_system["allowed_types_map"][t2]
                            if edge_type[1] != src and edge_type[1] != t1])
        return [("e0", r1, "?x0"), ("?x0", r2, "?x1"), ("?x1", r3, "?ans")], [("e0", src)]
    elif pattern_type == '2i':
        src1 = rng.choice(list(kb_system["allowed_types_map"].keys()))
        r1, t1 = rng.choice(kb_system["allowed_types_map"][src1])
        e2r2_cands = []
        for e, edge_types in kb_system["allowed_types_map"].items():
            if e == src1:
                continue
            for edge_type in edge_types:
                if edge_type[1] == t1:
                    e2r2_cands.append((e, edge_type[0]))
        src2, r2 = rng.choice(e2r2_cands)
        return [("e0", r1, "?ans"), ("e1", r2, "?ans")], [("e0", src1), ("e1", src2)]
    elif pattern_type == 'ip':
        src1 = rng.choice(list(kb_system["allowed_types_map"].keys()))
        r1, t1 = rng.choice(kb_system["allowed_types_map"][src1])
        e2r2_cands = []
        for e, edge_types in kb_system["allowed_types_map"].items():
            if e == src1:
                continue
            for edge_type in edge_types:
                if edge_type[1] == t1:
                    e2r2_cands.append((e, edge_type[0]))
        src2, r2 = rng.choice(e2r2_cands)
        r3, t3 = rng.choice([edge_type for edge_type in kb_system["allowed_types_map"][t1]
                            if edge_type[1] != src1 and edge_type[1] != src2])
        return [("e0", r1, "?x0"), ("e1", r2, "?x0"), ("?x0", r3, "?ans")], [("e0", src1), ("e1", src2)]
    elif pattern_type == 'pi':
        src1 = rng.choice(list(kb_system["allowed_types_map"].keys()))
        r1, t1 = rng.choice(kb_system["allowed_types_map"][src1])
        r2, t2 = rng.choice([edge_type for edge_type in kb_system["allowed_types_map"][t1] if edge_type[1] != src1])
        e3r3_cands = []
        for e, edge_types in kb_system["allowed_types_map"].items():
            if e == src1 or e == t1:
                continue
            for edge_type in edge_types:
                if edge_type[1] == t2:
                    e3r3_cands.append((e, edge_type[0]))
        src2, r3 = rng.choice(e3r3_cands)
        return [("e0", r1, "?x0"), ("?x0", r2, "?ans"), ("e1", r3, "?ans")], [("e0", src1), ("e1", src2)]


def get_graph(kb_system, pattern, seed_ents, max_size, max_depth, p, rng, prefix=""):
    graph = Graph(max_size + len(seed_ents))
    for se in seed_ents:
        graph.add_entity(prefix+se[0], se[1])
    graph.seed_entities(kb_system['entity_types'], prefix)
    expanded_arr = np.zeros(graph.size(), dtype=bool)
    for se in seed_ents:
        sample_graph(rng, kb_system, graph, graph.ent2id[prefix+se[0]], max_depth, p, expanded_arr)

    pattern_binding = insert_pattern_in_graph(graph, kb_system, pattern, rng)
    graph.prune_disconnected_entities()
    return graph, pattern_binding


def jsonify_adjacency_map(in_adj_map):
    out_adj_map = {}
    for (e1, r), e2_set in in_adj_map.items():
        if e1 not in out_adj_map:
            out_adj_map[e1] = {}
        assert r not in out_adj_map[e1]
        out_adj_map[e1][r] = list(e2_set)
    return out_adj_map


def reverse_jsonify_adjacency_map(in_adj_map):
    out_adj_map = {}
    for e1, r_dict in in_adj_map.items():
        for r, e2_list in r_dict.items():
            assert (e1, r) not in out_adj_map
            out_adj_map[(e1, r)] = set(e2_list)
    return out_adj_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb_system", type=str, default=None)
    parser.add_argument("--n_patterns", type=int, default=5)
    parser.add_argument("--ktr_per_pattern", type=int, default=5)
    parser.add_argument("--kval_per_pattern", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.4)
    parser.add_argument("--include_inv_edges", action='store_true')
    parser.add_argument("--max_graph_size", type=int, default=40)
    parser.add_argument("--max_graph_depth", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.kb_system is None:
        kb_system = {"entity_types": ENTITY_TYPES, "rel_types": REL_TYPES, "allowed_types_map": ALLOWED_TYPES_MAP}
    else:
        kb_system = json.load(open(args.kb_system))
    global_rng = np.random.default_rng(args.seed)

    train_ex_ctr, val_ex_ctr, test_ex_ctr = 0, 0, 0
    train_set, val_set, test_set = [], [], []
    for pat_id, pat in enumerate(global_rng.choice(PATTERN_TYPES, args.n_patterns, replace=True)):
        grounded_pattern, seed_entities = ground_pattern(pat, kb_system, global_rng)
        st_offset = len(train_set)
        for k_ in range(args.ktr_per_pattern):
            prefix = "train_"+str(train_ex_ctr)+"_"
            print(f'----\nSampling graph {prefix[:-1]}')
            # add prefix to pattern
            prefixed_pattern = []
            for (p_1, p_r, p_2) in grounded_pattern:
                if not p_1.startswith('?'):
                    p_1 = prefix + p_1
                if not p_2.startswith('?'):
                    p_2 = prefix + p_2
                prefixed_pattern.append((p_1, p_r, p_2))
            collected_graph, res = get_graph(kb_system, prefixed_pattern, seed_entities,
                                             args.max_graph_size, args.max_graph_depth, args.p, global_rng, prefix)
            train_set.append({"id": "train_"+str(train_ex_ctr),
                              "pattern_type": pat,
                              "grounded_pattern": prefixed_pattern,
                              "seed_entities": [prefix+se[0] for se in seed_entities],
                              "answer": res["?ans"],
                              "bindings": res,
                              "graph": {
                                  "entities": collected_graph.entities,
                                  "entity_types": collected_graph.ent_types,
                                  "adj_map": jsonify_adjacency_map(collected_graph.adj_map)}
                              })
            train_ex_ctr += 1
        en_offset = len(train_set)
        knn_range = ["train_"+str(i) for i in range(st_offset, en_offset)]
        for offset in range(st_offset, en_offset):
            train_set[offset]["knn"] = knn_range
        for k_ in range(args.kval_per_pattern):
            prefix = "valid_" + str(val_ex_ctr) + "_"
            print(f'----\nSampling graph {prefix[:-1]}')
            # add prefix to pattern
            prefixed_pattern = []
            for (p_1, p_r, p_2) in grounded_pattern:
                if not p_1.startswith('?'):
                    p_1 = prefix + p_1
                if not p_2.startswith('?'):
                    p_2 = prefix + p_2
                prefixed_pattern.append((p_1, p_r, p_2))
            collected_graph, res = get_graph(kb_system, prefixed_pattern, seed_entities,
                                             args.max_graph_size, args.max_graph_depth, args.p, global_rng, prefix)
            val_set.append({"id": "valid_" + str(val_ex_ctr),
                            "pattern_type": pat,
                            "grounded_pattern": prefixed_pattern,
                            "seed_entities": [prefix+se[0] for se in seed_entities],
                            "answer": res["?ans"],
                            "bindings": res,
                            "graph": {
                                "entities": collected_graph.entities,
                                "entity_types": collected_graph.ent_types,
                                "adj_map": jsonify_adjacency_map(collected_graph.adj_map)},
                            "knn": knn_range
                            })
            val_ex_ctr += 1
        for k_ in range(args.kval_per_pattern):
            prefix = "test_" + str(test_ex_ctr) + "_"
            print(f'----\nSampling graph {prefix[:-1]}')
            # add prefix to pattern
            prefixed_pattern = []
            for (p_1, p_r, p_2) in grounded_pattern:
                if not p_1.startswith('?'):
                    p_1 = prefix + p_1
                if not p_2.startswith('?'):
                    p_2 = prefix + p_2
                prefixed_pattern.append((p_1, p_r, p_2))
            collected_graph, res = get_graph(kb_system, prefixed_pattern, seed_entities,
                                             args.max_graph_size, args.max_graph_depth, args.p, global_rng, prefix)
            test_set.append({"id": "test_" + str(test_ex_ctr),
                             "pattern_type": pat,
                             "grounded_pattern": prefixed_pattern,
                             "seed_entities": [prefix+se[0] for se in seed_entities],
                             "answer": res["?ans"],
                             "bindings": res,
                             "graph": {
                                 "entities": collected_graph.entities,
                                 "entity_types": collected_graph.ent_types,
                                 "adj_map": jsonify_adjacency_map(collected_graph.adj_map)},
                             "knn": knn_range
                             })
            test_ex_ctr += 1
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.kb_system is not None:
        out_filenm = os.path.join(args.output_dir, os.path.split(args.kb_system)[-1])
        shutil.copyfile(args.kb_system, out_filenm)
    with open(os.path.join(args.output_dir, "train.json"), 'w') as fout:
        json.dump(train_set, fout, indent=2)
    with open(os.path.join(args.output_dir, "dev.json"), 'w') as fout:
        json.dump(val_set, fout, indent=2)
    with open(os.path.join(args.output_dir, "test.json"), 'w') as fout:
        json.dump(test_set, fout, indent=2)
    with open(os.path.join(args.output_dir, "dset_sampling_config.json"), 'w') as fout:
        json.dump(vars(args), fout, indent=4)
