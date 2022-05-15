import json
from typing import *
from random import randrange
import urllib.parse
import urllib.request
import http
import re
from collections import defaultdict
from tqdm import tqdm
from numpy.random import default_rng

rng = default_rng()


def sparqlQuery(query, baseURL, format="application/json"):
    params = {
        "default-graph": "",
        "should-sponge": "soft",
        "query": query,
        "debug": "on",
        "timeout": "",
        "format": format,
        "save": "display",
        "fname": ""
    }
    querypart = urllib.parse.urlencode(params).encode("utf-8")
    response = urllib.request.urlopen(baseURL, querypart).read()
    return json.loads(response)


def execute_kb_query(query: str) -> Tuple[List[str], bool]:
    PREFIX = "PREFIX ns: <http://rdf.freebase.com/ns/> "
    query = PREFIX + query
    is_exception = False
    ans_entities = []
    error_msg = ""
    urls = ["http://gypsum.cs.umass.edu:3001/sparql/"]
    try:
        # select a url randomly from the list of services
        url = urls[randrange(len(urls))]
        data = sparqlQuery(query, url)

        if "results" in data:
            if "bindings" in data["results"]:
                bindings = data["results"]["bindings"]
                for b in bindings:
                    for var in b.keys():
                        if "value" in b[var]:
                            ans_entity = b[var]["value"]
                            # parse the url
                            ans_entity = ans_entity.split("/")[-1]
                            ans_entities.append(ans_entity)
    except urllib.error.HTTPError:
        is_exception = True
        error_msg = "HTTPError exception. Might be syntax."
    except urllib.error.URLError:
        is_exception = True
        error_msg = "URLError. Service might not be responding"
    except http.client.RemoteDisconnected:
        is_exception = True
        error_msg = "URLError. Service might not be responding"

    return ans_entities, is_exception, error_msg


def execute_kb_query_for_hops(query: str, hop=2) -> Tuple[List[str], bool]:
    PREFIX = "PREFIX ns: <http://rdf.freebase.com/ns/> "
    query = PREFIX + query
    is_exception = False
    ans_entities = set()
    error_msg = ""
    urls = ["http://gypsum.cs.umass.edu:3001/sparql/"]
    try:
        # select a url randomly from the list of services
        url = urls[randrange(len(urls))]
        data = sparqlQuery(query, url)
        if "results" in data:
            if "bindings" in data["results"]:
                bindings = data["results"]["bindings"]
                for b in bindings:
                    r1, r2, r3 = None, None, None
                    for var in b.keys():
                        if "value" in b[var]:
                            if var == "r1":
                                r1 = b[var]["value"]
                                r1 = r1.split("/")[-1]
                            if var == "r2":
                                r2 = b[var]["value"]
                                r2 = r2.split("/")[-1]
                            if var == "r3":
                                r3 = b[var]["value"]
                                r3 = r3.split("/")[-1]
                    if hop == 3 and r1 is not None and r2 is not None and r3 is not None:
                        ans_entities.add((r1, r2, r3))
                    if hop == 2 and r1 is not None and r2 is not None:
                        ans_entities.add((r1, r2))
                    if hop == 1 and r1 is not None:
                        ans_entities.add((r1))

    except urllib.error.HTTPError as e:
        is_exception = True
        error_msg = e.__str__()
    except urllib.error.URLError:
        is_exception = True
        error_msg = "URLError. Service might not be responding"
    except http.client.RemoteDisconnected:
        is_exception = True
        error_msg = "URLError. Service might not be responding"

    return ans_entities, is_exception, error_msg


def get_query_entities_and_answers_cwq(input_file, return_gold_entities=False):
    qid2qents, qid2answers, qid2gold_chains, qid2q_str, qid2gold_spqls = {}, {}, {}, {}, {}
    with open(input_file) as fin:
        data = json.load(fin)
    for d in data:
        qid = d["ID"]
        if return_gold_entities:
            all_query_entities = set()
            if "sparql" in d:
                spql = d["sparql"]
                sub_str = "ns:m\."
                start_indxs = [m.start() for m in re.finditer(sub_str, spql)]
                for st_ind in start_indxs:
                    curr_ind = st_ind
                    while (spql[curr_ind] != ' ') and (spql[curr_ind] != ')'):
                        curr_ind += 1
                    en_ind = curr_ind
                    all_query_entities.add(spql[st_ind + 3:en_ind])
        else:
            all_query_entities = set([m[0] for m in d["mentions"]])
        qid2qents[qid] = all_query_entities
        qid2q_str[qid] = d["question"]
        all_spqls = [d["sparql"]]
        qid2gold_spqls[qid] = all_spqls
        answers = [a["answer_id"] for a in d["answers"]] if "answers" in d else []
        qid2answers[qid] = answers
    return qid2qents, qid2answers, qid2gold_spqls, qid2q_str


def get_query_entities_and_answers(input_file, return_gold_entities=False):
    qid2qents, qid2answers, qid2gold_chains, qid2q_str = {}, {}, {}, {}
    with open(input_file) as fin:
        data = json.load(fin)
    for d in data["Questions"]:
        qid = d["QuestionId"]
        qid2q_str[qid] = d["ProcessedQuestion"]
        ans_ids = set()
        all_inference_chains = set()
        all_query_entities = set()
        if 'Parses' in d:
            for p in d['Parses']:
                if return_gold_entities:
                    if 'TopicEntityMid' in p:
                        if p['TopicEntityMid'] is not None:
                            all_query_entities.add(p['TopicEntityMid'])
                    # also get entities from constraints
                    if 'Constraints' in p:
                        constraints = p['Constraints']
                        for c in constraints:
                            if c["ArgumentType"] == "Entity":
                                if c["Argument"] is not None:
                                    all_query_entities.add(c["Argument"])
                answers = p["Answers"]
                for a in answers:
                    if a["AnswerArgument"] is not None:
                        ans_ids.add(a["AnswerArgument"])
                inference_chain = tuple(p["InferentialChain"]) if p["InferentialChain"] is not None else None
                all_inference_chains.add(inference_chain)
        if not return_gold_entities:
            assert len(all_query_entities) == 0
            all_query_entities = set([m[0] for m in d["mentions"]])
        qid2qents[qid] = all_query_entities
        qid2answers[qid] = ans_ids
        qid2gold_chains[qid] = all_inference_chains
    return qid2qents, qid2answers, qid2gold_chains, qid2q_str


def get_query_entities_and_answers_freebaseqa(input_file, return_gold_entities=True):
    qid2qents, qid2answers, qid2gold_chains, qid2q_str = {}, {}, {}, {}
    with open(input_file) as fin:
        data = json.load(fin)
    for d in data["Questions"]:
        qid = d["Question-ID"]
        qid2q_str[qid] = d["ProcessedQuestion"]
        ans_ids = set()
        all_inference_chains = set()
        all_query_entities = set()
        if 'Parses' in d:
            for p in d['Parses']:
                if 'TopicEntityMid' in p:
                    if p['TopicEntityMid'] is not None:
                        all_query_entities.add(p['TopicEntityMid'])
                answers = p["Answers"]
                for a in answers:
                    if a["AnswersMid"] is not None:
                        ans_ids.add(a["AnswersMid"])
                inference_chain = tuple(p["InferentialChain"].split("..")) if p[
                                                                                  "InferentialChain"] is not None else None
                all_inference_chains.add(inference_chain)
        qid2qents[qid] = all_query_entities
        qid2answers[qid] = ans_ids
        qid2gold_chains[qid] = all_inference_chains
    return qid2qents, qid2answers, qid2gold_chains, qid2q_str


def get_query_entities_and_answers_metaqa(input_file, return_gold_entities=True):
    assert return_gold_entities is True  # metaqa comes with tagged entities
    qid2qents, qid2answers, qid2gold_chains, qid2q_str = {}, {}, {}, {}
    with open(input_file) as fin:
        data = json.load(fin)
    for d in data:
        qid = d["id"]
        qid2q_str[qid] = d["question"]
        qid2qents[qid] = d["seed_entities"]
        qid2gold_chains[qid] = None
        qid2answers[qid] = d["answer"]
    return qid2qents, qid2answers, qid2gold_chains, qid2q_str


def read_metaqa_kb(kb_file):
    e1_map = defaultdict(list)
    with open(kb_file) as fin:
        for line in tqdm(fin):
            line = line.strip()
            e1, r, e2 = line.split("|")
            e1_map[e1].append((r, e2))
            e1_map[e2].append((r+"_inv", e1))
    return e1_map

def read_metaqa_kb_for_traversal(kb_file):
    e1_r_map = defaultdict(list)
    with open(kb_file) as fin:
        for line in tqdm(fin):
            line = line.strip()
            e1, r, e2 = line.split("|")
            e1_r_map[(e1, r)].append(e2)
            e1_r_map[(e2, r+"_inv")].append(e1)
    return e1_r_map


def find_paths(e1_map, q_ent, ans_ent, num_max_paths=1000, max_path_len=3):
    all_collected_paths = set()
    for _ in range(num_max_paths):
        prefix_rel = []
        prefix_ent = set()
        curr_ent = q_ent
        for l in range(max_path_len):
            outgoing_edges = e1_map[curr_ent]
            if len(outgoing_edges) == 0:
                break
            r, e2 = rng.choice(outgoing_edges)
            if e2 in prefix_ent:
                # this is a loopy path as it ends on an ent seen before. Ignore this
                break
            prefix_rel.append(r)
            prefix_ent.add(e2)
            if e2 == ans_ent:
                all_collected_paths.add(tuple(prefix_rel))
            curr_ent = e2
    return all_collected_paths