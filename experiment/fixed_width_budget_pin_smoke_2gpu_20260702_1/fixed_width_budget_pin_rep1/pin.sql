CALL pin_table('/mnt/nvme/dataset/customer*.parquet', tier='gpu', name='customer', cols=['c_custkey']);
CALL pin_table('/mnt/nvme/dataset/lineitem*.parquet', tier='gpu', name='lineitem', cols=['l_orderkey']);
CALL pin_table('/mnt/nvme/dataset/orders*.parquet', tier='gpu', name='orders', cols=['o_custkey','o_orderdate','o_orderkey']);
