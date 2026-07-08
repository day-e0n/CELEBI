CALL pin_table('/mnt/nvme/dataset/lineitem*.parquet', tier='gpu', name='lineitem', cols=['l_discount','l_quantity','l_shipdate']);
