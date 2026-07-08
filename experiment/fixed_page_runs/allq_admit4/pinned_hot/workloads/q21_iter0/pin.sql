CALL pin_table('/mnt/nvme/dataset/lineitem*.parquet', tier='gpu', name='lineitem', cols=['l_orderkey']);
CALL pin_table('/mnt/nvme/dataset/nation*.parquet', tier='gpu', name='nation', cols=['n_nationkey']);
CALL pin_table('/mnt/nvme/dataset/orders*.parquet', tier='gpu', name='orders', cols=['o_orderkey']);
CALL pin_table('/mnt/nvme/dataset/supplier*.parquet', tier='gpu', name='supplier', cols=['s_nationkey','s_suppkey']);
