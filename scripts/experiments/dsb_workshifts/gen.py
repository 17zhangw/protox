import copy
from pathlib import Path


if __name__ == "__main__":
    def _order(x):
        x = str(x)
        if "dsb_pd" in x:
            return (1, x)
        elif "dsb_newparams" in x:
            return (2, x)
        elif "dsb_ut" in x:
            return (3, x)
        return (100, x)
    qorders = Path("/home/wz2/mythril/queries/specializations/").rglob("d_qorder.txt")
    qorders = [q for q in qorders if "dsb_ut" in str(q) or "dsb_pd" in str(q) or "dsb_newparams_3" in str(q) or "dsb_s2_0" in str(q) or "dsb_s3_0" in str(q)]
    qorders = sorted(qorders, key=_order)

    history_pths = [
        "/nfs1/exact_transfer/unitune/dsb_exact/dsb_exact_0.5492/tpch_test.res",
        "/nfs1/exact_transfer/unitune/dsb_exact/dsb_exact_0.5493/tpch_test.res",
        "/nfs1/exact_transfer/unitune/dsb_exact/dsb_exact_0.5496/tpch_test.res",
        "/nfs1/exact_transfer/unitune/dsb_exact/dsb_exact_0.5497/tpch_test.res",
    ]

    with open("uhm.template", "r") as f:
        template = f.read()

    ports = [5492, 5493, 5494, 5495, 5496, 5497]
    ppcmds = [[], [], [], [], [], []]
    
    for i, qorder in enumerate(qorders):
        print(qorder)
        qname = qorder.parts[-2]

        for it in range(4):
            slot = i*4 + it
            sidx = slot % len(ports)
            port = ports[sidx]

            tt = copy.deepcopy(template)
            tt = tt.replace("logs5450", f"logs{port}")
            tt = tt.replace("port = 5450", f"port = {port}")

            # Replace the workload_qlist_file.
            tt = tt.replace(
                "/home/wz2/mythril/queries/dsb_revise/d_qorder.txt",
                str(qorder),
            )

            tt = tt.replace(
                "history_load = None",
                f"history_load = {history_pths[it]}",
            )

            tt = tt.replace(
                "/home/wz2/mythril/queries/dsb_revise/",
                str(qorder.parent),
            )

            tt = tt.replace(
                "tuning_budget = 108000",
                "tuning_budget = 43200",
            )

            with open(f"{qorder.parent.stem}_{it}.special", "w") as f:
                f.write(tt)

            ppcmds[sidx].append(
                f"PORT={port} NAME={qname}_{it} ARCHIVE=/home/wz2/mythril/data/dsb_sf10.tgz CONFIG={qorder.parent.stem}_{it}.special ./scripts/experiments/dsb_workshifts/unitune0.sh"
            )

    for i, ppcmd in enumerate(ppcmds):
        with open(f"/home/wz2/mythril/u{i}.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -ex\n")
            f.write("set -o pipefail\n")
            f.write("\n".join(ppcmd))
