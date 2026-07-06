CALL pin_table('/mnt/nvme/dataset/customer*.parquet', tier='gpu', name='customer', cols=['c_custkey','c_nationkey']);
CALL pin_table('/mnt/nvme/dataset/lineitem*.parquet', tier='gpu', name='lineitem', cols=['l_discount','l_extendedprice','l_orderkey','l_shipdate','l_suppkey']);
CALL pin_table('/mnt/nvme/dataset/nation*.parquet', tier='gpu', name='nation', cols=['n_nationkey','n_regionkey']);
CALL pin_table('/mnt/nvme/dataset/orders*.parquet', tier='gpu', name='orders', cols=['o_custkey','o_orderdate','o_orderkey']);
CALL pin_table('/mnt/nvme/dataset/region*.parquet', tier='gpu', name='region', cols=['r_regionkey']);
CALL pin_table('/mnt/nvme/dataset/supplier*.parquet', tier='gpu', name='supplier', cols=['s_nationkey','s_suppkey']);
