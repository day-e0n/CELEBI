# Fixed-column sequence workload 3x

Queries: `q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22`

Conditions: `cold_hot,paging`

Repeat layout: `query`

Fixed-width page bytes: `2097152`

Series rule: cold=cold_hot execution 1, hot=cold_hot executions 2-3, paging=paging executions 2-3.
