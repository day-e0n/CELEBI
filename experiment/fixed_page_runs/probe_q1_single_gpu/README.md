# Fixed-column sequence workload 3x

Queries: `q1`

Conditions: `cold_hot,paging`

Repeat layout: `query`

Fixed-width page bytes: `2097152`

Series rule: cold=cold_hot execution 1, hot=cold_hot executions 2-3, paging=paging executions 2-3.
