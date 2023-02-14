mythril:
  benchmark: tpch
  oltp_workload: False

  query_spec:
    benchbase: False
    allow_per_query: True
    early_workload_kill: True
    query_directory: "/home/wz2/mythril/queries/tpch"
    query_order: "/home/wz2/mythril/queries/tpch/order_bao.txt"
    tbl_include_subsets_prune: True

  max_num_columns: 16
  tables:
    - part
    - partsupp
    - lineitem
    - orders
    - supplier
    - customer
    - nation
    - region

  attributes:
    region:
      - r_regionkey
      - r_name
      - r_comment
    nation:
      - n_nationkey
      - n_name
      - n_regionkey
      - n_comment
    part:
      - p_partkey
      - p_name
      - p_mfgr
      - p_brand
      - p_type
      - p_size
      - p_container
      - p_retailprice
      - p_comment
    supplier:
      - s_suppkey
      - s_name
      - s_address
      - s_nationkey
      - s_phone
      - s_acctbal
      - s_comment
    partsupp:
      - ps_partkey
      - ps_suppkey
      - ps_availqty
      - ps_supplycost
      - ps_comment
    customer:
      - c_custkey
      - c_name
      - c_address
      - c_nationkey
      - c_phone
      - c_acctbal
      - c_mktsegment
      - c_comment
    orders:
      - o_orderkey
      - o_custkey
      - o_orderstatus
      - o_totalprice
      - o_orderdate
      - o_orderpriority
      - o_clerk
      - o_shippriority
      - o_comment
    lineitem:
      - l_orderkey
      - l_partkey
      - l_suppkey
      - l_linenumber
      - l_quantity
      - l_extendedprice
      - l_discount
      - l_tax
      - l_returnflag
      - l_linestatus
      - l_shipdate
      - l_commitdate
      - l_receiptdate
      - l_shipinstruct
      - l_shipmode
      - l_comment

  # Additional table level knobs.
  # Format:
  #   <tbl_name>:
  #     <Knob Specification 0>
  #     <Knob Specification 1>
  #     ...
  table_level_knobs: {}

  # Per-query knobs.
  # Format:
  #   <benchbase TransactionType.name>:
  #     <Knob Specification 0>
  #     ...
  per_query_scan_method: False
  per_query_select_parallel: False
  index_space_aux_type: False
  index_space_aux_include: False

  per_query_knob_gen:
      enable_seqscan:   {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_indexscan:   {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_hashjoin:    {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_mergejoin:   {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_nestloop:    {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_sort:                            {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_gathermerge:                     {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_hashagg:                         {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_parallel_hash:                   {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_material:                        {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}
      enable_memoize:                         {type: "boolean", min: 0, max: 1, quantize: 0, log_scale: 0, unit: 0}

  per_query_knobs: {}
