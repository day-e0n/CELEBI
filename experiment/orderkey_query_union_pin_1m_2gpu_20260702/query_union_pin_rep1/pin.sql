CALL pin_table('/mnt/nvme/dataset/customer*.parquet', tier='gpu', name='customer', cols=['c_acctbal','c_address','c_comment','c_custkey','c_mktsegment','c_name','c_nationkey','c_phone'], n_rows=1000000);
CALL pin_table('/mnt/nvme/dataset/lineitem*.parquet', tier='gpu', name='lineitem', cols=['l_commitdate','l_discount','l_extendedprice','l_orderkey','l_partkey','l_quantity','l_receiptdate','l_returnflag','l_shipdate','l_suppkey'], n_rows=1000000);
CALL pin_table('/mnt/nvme/dataset/nation*.parquet', tier='gpu', name='nation', cols=['n_name','n_nationkey','n_regionkey'], n_rows=1000000);
CALL pin_table('/mnt/nvme/dataset/orders*.parquet', tier='gpu', name='orders', cols=['o_custkey','o_orderdate','o_orderkey','o_orderstatus','o_shippriority','o_totalprice'], n_rows=1000000);
CALL pin_table('/mnt/nvme/dataset/part*.parquet', tier='gpu', name='part', cols=['p_name','p_partkey','p_type'], n_rows=1000000);
CALL pin_table('/mnt/nvme/dataset/partsupp*.parquet', tier='gpu', name='partsupp', cols=['ps_partkey','ps_suppkey','ps_supplycost'], n_rows=1000000);
CALL pin_table('/mnt/nvme/dataset/region*.parquet', tier='gpu', name='region', cols=['r_name','r_regionkey'], n_rows=1000000);
CALL pin_table('/mnt/nvme/dataset/supplier*.parquet', tier='gpu', name='supplier', cols=['s_name','s_nationkey','s_suppkey'], n_rows=1000000);
