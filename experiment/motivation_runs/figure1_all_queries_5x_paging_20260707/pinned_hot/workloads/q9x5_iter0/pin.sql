CALL pin_table('/mnt/nvme/dataset/lineitem*.parquet', tier='gpu', name='lineitem', cols=['l_orderkey']);
CALL pin_table('/mnt/nvme/dataset/nation*.parquet', tier='gpu', name='nation', cols=['n_nationkey']);
CALL pin_table('/mnt/nvme/dataset/orders*.parquet', tier='gpu', name='orders', cols=['o_orderkey']);
CALL pin_table('/mnt/nvme/dataset/part*.parquet', tier='gpu', name='part', cols=['p_partkey']);
CALL pin_table('/mnt/nvme/dataset/partsupp*.parquet', tier='gpu', name='partsupp', cols=['ps_partkey','ps_suppkey','ps_supplycost']);
CALL pin_table('/mnt/nvme/dataset/supplier*.parquet', tier='gpu', name='supplier', cols=['s_nationkey','s_suppkey']);
