# Stage retention matrices

Each subdirectory contains 22x22 matrices for:

- retention cost GB
- potential reuse loss GB
- retention efficiency = reuse_loss / retained_output_bytes

Diagonal cells and queries without the selected stage are blank.

Stages:

- `filter/`: 9 queries with output, max q21=18.588473GB
- `join/`: 18 queries with output, max q9=7.423850GB
- `aggregate/`: 20 queries with output, max q18=7.219595GB
- `partition/`: 19 queries with output, max q9=38.856955GB
- `concat/`: 18 queries with output, max q9=38.849681GB
- `projection/`: 20 queries with output, max q1=37.252112GB
- `sort/`: 12 queries with output, max q16=0.005418GB
- `limit/`: 4 queries with output, max q2=0.000036GB
