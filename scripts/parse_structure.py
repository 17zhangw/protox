import json
import tqdm
import argparse
from pathlib import Path


def count_indent(line):
    i = 0
    while line[i] == ' ':
        i += 1

    # i is number of whitespace.
    # 2 is the minor indent.
    indent = (-1) if i == 0 else (i - 2) / 6
    return indent

def plan_repr(plan):
    node = plan["Node"]
    nodes = []
    for k, v in plan.items():
        if k == "Plans":
            for vv in v:
                nodes.append(plan_repr(vv))
        elif k == "Workers Planned":
            node = node + f" {v}"

    if len(nodes) == 0:
        return node

    if len(nodes) == 1:
        if isinstance(nodes[0], tuple):
            return (node, nodes[0])

    return (node, tuple(nodes))

def parse_plan(qid, flags, plan):
    current_level = {}
    current_indent = count_indent(plan[0])

    line = plan[0].strip()
    assert line.startswith("-> ") or current_indent == -1

    # Starting a new level...
    splits = line.split("-> ")[-1].split(" (", 1)
    current_level["Node"] = splits[0].strip()

    splits = splits[1].split(") (", 1)
    ests = splits[0].split(" ")
    current_level["Cost"] = ests[0].split("=")[-1]
    current_level["Rows"] = ests[1].split("=")[-1]
    current_level["Width"] = ests[2].split("=")[-1]

    if len(splits) > 1:
        if "never executed" in splits[1]:
            current_level["Skipped"] = True
        else:
            acts = splits[1].split(")")[0]
            acts = acts.split(" loops")
            current_level["Actual Rows"] = acts[0].split("=")[-1]
            current_level["Actual Loops"] = acts[1].split("=")[-1]

    plan = plan[1:]
    while len(plan) > 0:
        line = plan[0]

        if line.strip().startswith("-> "):
            next_indent = count_indent(line)
            if next_indent == current_indent + 1:
                if "Plans" not in current_level:
                    current_level["Plans"] = []
                child, plan = parse_plan(qid, flags, plan)
                current_level["Plans"].append(child)
            elif next_indent <= current_indent:
                return current_level, plan
            else:
                assert False
        else:
            k, v = line.split(": ", 1)
            current_level[k.strip()] = v.strip()
            plan = plan[1:]

    return current_level, plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Parser")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    flags = ["PrevDual", "GlobalDual", "PerQuery", "PerQueryInverse"]

    num_lines = 0
    with open(args.input, "r") as f:
        for line in f:
            num_lines += 1

    parsed_plans = []

    with open(args.input, "r") as f:
        current_query = None
        current_flags = None
        current_plan = []

        for line in f: #tqdm.tqdm(f, total=num_lines):
            if line.startswith("Q"):
                if current_query is not None:
                    plan, _ = parse_plan(current_query, current_flags, current_plan)
                    parsed_plans.append((current_query, plan_repr(plan)))

                    current_query = None
                    current_flags = None
                    current_plan = []

                current_query = line
                continue
            elif any([line.startswith(f) for f in flags]):
                assert current_query is not None
                current_flags = line.split(": ")
                current_flags = (current_flags[0], eval(current_flags[1]))
                continue
            elif len(line.strip()) == 0:
                continue

            current_plan.append(line)

        if current_query is not None:
            parsed_plans.append((current_query, plan_repr(plan)))

    with open(args.output, "w") as f:
        for (qid, plan) in parsed_plans:
            qid = qid.strip()
            f.write(f"{qid}\n")
            f.write(f"{plan}\n\n")
