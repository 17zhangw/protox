import copy
import pandas as pd
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

#workshifts_data_unitune_results/
#dsb_newparams_3_sf10_wl_0_baseline_norm_k5.csv
#dsb_pd10_sf10_wl_0_baseline_norm_k5.csv
#dsb_pd25_sf10_wl_0_baseline_norm_k5.csv
#dsb_pd50_sf10_wl_0_baseline_norm_k5.csv
#dsb_pd75_sf10_wl_0_baseline_norm_k5.csv
#dsb_pd90_sf10_wl_0_baseline_norm_k5.csv
#dsb_s2_0_sf10_wl_0_baseline_norm_k5.csv
#dsb_s3_0_sf10_wl_0_baseline_norm_k5.csv
#dsb_ut100_sf10_wl_0_baseline_norm_k5.csv
#dsb_ut10_sf10_wl_0_baseline_norm_k5.csv
#dsb_ut25_sf10_wl_0_baseline_norm_k5.csv
#dsb_ut50_sf10_wl_0_baseline_norm_k5.csv
#dsb_ut75_sf10_wl_0_baseline_norm_k5.csv
#dsb_ut90_sf10_wl_0_baseline_norm_k5.csv

    with open("uhm.template", "r") as f:
        template = f.read()

    ports = [5492, 5493, 5494, 5495, 5496, 5497]
    ppcmds = [[], [], [], [], [], []]
    
    for i, qorder in enumerate(qorders):
        print(qorder)
        qname = qorder.parts[-2]
        norm = f"/nfs1/workshifts_data_unitune_results/{qname}_sf10_wl_0_baseline_norm_k5.csv"
        assert Path(norm).exists()
        df = pd.read_csv(norm).drop(columns=["step", "time_since_start"]).set_index(keys=["base_input"], drop=True).min(axis=1)
        norm = "best-{},300".format(df[df == df.min()].index[0])

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
                f"history_load = {norm}",
            )

            tt = tt.replace(
                "/home/wz2/mythril/queries/dsb_revise/",
                str(qorder.parent),
            )

            tt = tt.replace(
                "tuning_budget = 108000",
                "tuning_budget = 43200",
            )

            with open(f"{qorder.parent.stem}_wmap_{it}.special", "w") as f:
                f.write(tt)

            ppcmds[sidx].append(
                f"PORT={port} NAME={qname}_wmap_{it} ARCHIVE=/home/wz2/mythril/data/dsb_sf10.tgz CONFIG={qorder.parent.stem}_wmap_{it}.special ./scripts/experiments/dsb_workshifts/unitune0.sh"
            )

    for i, ppcmd in enumerate(ppcmds):
        with open(f"/home/wz2/mythril/u{i}.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -ex\n")
            f.write("set -o pipefail\n")
            f.write("\n".join(ppcmd))
