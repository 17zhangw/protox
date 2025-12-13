from dateutil.parser import parse


def scan_index_str(index_str):
    indexes = set()
    current_index = None
    current_column = None
    for i in range(len(index_str)):
        if index_str[i] == "I":
            current_index = []
        elif index_str[i] == "C":
            current_column = ""
        elif index_str[i] == ",":
            if current_index is not None:
                current_index.append(current_column)
                current_column = ""
        elif index_str[i] == ")":
            if current_column is not None and len(current_column) > 0:
                current_index.append(current_column)
            indexes.add(tuple(current_index))
            current_index = None
        elif current_column is not None and index_str[i] != " ":
            current_column += index_str[i]

    indexdefs = []
    indexas = set()
    for index_cols in indexes:
        index_tbl = None
        index_colnames = []
        for ic in index_cols:
            tbl, col = ic.split(".")
            if index_tbl is None:
                index_tbl = tbl
            index_colnames.append(col)

        cnames = ",".join(index_colnames)
        indexdefs.append(f"CREATE INDEX index{len(indexdefs)} ON {index_tbl} ({cnames})")
        indexas.add((index_tbl, cnames))
    return indexdefs, indexas


def parse_hyrise_configs(args):
    nindexdefs = []
    nindexas = []
    ntimes = []
    start = None
    with open(args.hyrise_log) as f:
        for line in f:
            if start is None:
                start = parse(line.split("]")[0][1:])
            if "Indexes found: " in line:
                indexstr = line.split("Indexes found: ")[-1]
                indexdefs, indexas = scan_index_str(indexstr)
                if not any([nidx == indexas for nidx in nindexas]):
                    tstart = line.split("Indexes found: ")[0].rfind("[")
                    time = line[tstart:].split("]")
                    ntimes.append((parse(time) - start).total_seconds())

                    nindexdefs.append(indexdefs)
                    nindexas.append(indexas)
            elif "AnytimeDTA found new best:" in line:
                last_dta = line.strip()
                last_dta = last_dta.split("AnytimeDTA found new best: ")[-1]
                indexdefs, indexas = scan_index_str(last_dta)
                if not any([nidx == indexas for nidx in nindexas]):
                    tstart = line.split("AnytimeDTA found new best: ")[0].rfind("[")
                    time = line[tstart+1:].split("]")[0]
                    ntimes.append((parse(time) - start).total_seconds())

                    nindexdefs.append(indexdefs)
                    nindexas.append(indexas)

    if len(nindexdefs) == 0:
        with open(args.hyrise_log) as f:
            addtl_indexes = []
            for line in f:
                if "Additional best index found: " in line:
                    addtl_indexes.append(line)

            # Attempt to read the stream of additional best indexes found.
            indexdefs = []
            for addtl_index in addtl_indexes:
                indexdef = addtl_index.split("(I(")[-1].split("), ")[0]
                cols = indexdef.split(",")
                columns = []
                table = None
                for col in cols:
                    assert col.startswith("C ")
                    coldef = col[2:].split(".")
                    if table is None:
                        table = coldef[0]
                    else:
                        assert table == coldef[0]
                    columns.append(coldef[1])

                cnames = ",".join(columns)
                indexdefs.append(f"CREATE INDEX index{len(indexdefs)} ON {table} ({cnames})")
            nindexdefs.append(indexdefs)

            line = addtl_indexes[-1]
            tstart = line.split("Additional best index found: ")[0].rfind("[")
            time = line[tstart:].split("]")
            ntimes.append((parse(time) - start).total_seconds())
    return nindexdefs, ntimes
