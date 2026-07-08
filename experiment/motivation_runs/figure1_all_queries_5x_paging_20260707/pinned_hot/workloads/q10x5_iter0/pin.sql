CALL pin_table('/mnt/nvme/dataset/customer*.parquet', tier='gpu', name='customer', cols=['c_custkey','c_nationkey']);
CALL pin_table('/mnt/nvme/dataset/lineitem*.parquet', tier='gpu', name='lineitem', cols=['l_orderkey']);
CALL pin_table('/mnt/nvme/dataset/nation*.parquet', tier='gpu', name='nation', cols=['n_nationkey']);
CALL pin_table('/mnt/nvme/dataset/orders*.parquet', tier='gpu', name='orders', cols=['o_custkey','o_orderkey']);
