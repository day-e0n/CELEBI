CALL pin_table('/mnt/nvme/dataset/part*.parquet', tier='gpu', name='part', cols=['p_partkey']);
CALL pin_table('/mnt/nvme/dataset/partsupp*.parquet', tier='gpu', name='partsupp', cols=['ps_partkey','ps_suppkey']);
CALL pin_table('/mnt/nvme/dataset/supplier*.parquet', tier='gpu', name='supplier', cols=['s_suppkey']);
