"""One agent sets the areas, one site NLP solves the operation (Lo et al. 2021 model).

The agent fixes every discrete and capital decision (stage of each existing unit, its
definite area, the definite area of each candidate intermediate-fluid exchanger, the loop
flow and its split).  A single Pyomo/IPOPT model of the whole site (the paper's stage-wise
equations A17-A36, A76 and the loop closure) then solves the operating state: duties,
branch temperatures, split and bypass fractions, utilities, loop temperatures.  Costs
and the paper's Sec. 6 rules are evaluated on the solution.
"""
