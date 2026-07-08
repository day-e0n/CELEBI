CALL pin_table('/mnt/nvme/dataset/part*.parquet', tier='gpu', name='part', cols=['p_partkey']);
