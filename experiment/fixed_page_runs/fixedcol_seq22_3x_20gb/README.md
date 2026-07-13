# Fixed-column sequence workload 3x

Queries: `q3,q10,q7,q5,q8,q9,q20,q11,q2,q16,q19,q17,q14,q1,q6,q15,q21,q12,q4,q18,q13,q22`

Conditions: `cold_hot,paging`

Series rule: cold=cold_hot execution 1, hot=cold_hot executions 2-3, paging=paging executions 2-3.
