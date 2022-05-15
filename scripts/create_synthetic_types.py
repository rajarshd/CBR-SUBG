import argparse
import json
import numpy as np


def sample_kb_system(n_etypes, p, seed, out_filenm):
    rng = np.random.default_rng(seed)
    ent_types = [f"t{i}" for i in range(n_etypes)]
    rel_types = []
    allowed_types_map = {}
    for i in range(n_etypes):
        for j in range(n_etypes):
            if i == j:
                continue
            if rng.random() < p:
                new_rel_name = f"r.t{i}.t{j}"
                rel_types.append(new_rel_name)
                if f"t{i}" not in allowed_types_map:
                    allowed_types_map[f"t{i}"] = [(new_rel_name, f"t{j}")]
                else:
                    allowed_types_map[f"t{i}"].append((new_rel_name, f"t{j}"))
    with open(out_filenm, 'w') as fout:
        json.dump({"entity_types": ent_types,
                   "rel_types": rel_types,
                   "allowed_types_map": allowed_types_map,
                   "config": {"n_etypes": n_etypes, "p": p, "seed": seed}},
                  fout, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_etypes", type=int, default=20)
    parser.add_argument("--p", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    if args.outfile is None:
        args.outfile = f"kb_system_{args.n_etypes}_{args.p}_{args.seed}.json"

    sample_kb_system(args.n_etypes, args.p, args.seed, args.outfile)
