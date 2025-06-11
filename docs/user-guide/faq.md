
todo: general update
todo: add structure
todo: more questions
# Frequently Asked Questions

## Category

### Does PyPSA model more than just the power system? E.g. methane, hydrogen, carbon dioxide networks?
Yes, PyPSA can model any energy carrier or material flow. Typically this is done as a transport network with linear losses. Example: PyPSA-Eur gas networks. Fabian Hofmann’s [H2 and CO2 network paper](https://arxiv.org/abs/2402.19042).
On the demand side, PyPSA has been used to model hte full energy system, including building heating (heat pumps, gas boilers, district heating), process heating in industry, process emissions in industry, electric vehicles with flexible (dis)charging, demands for transport fuels and industrial feedstocks.

### How is demand modelled?
Demand can be modelled with linear or convex-quadratic models: perfectly inelastic, perfectly inelastic up to a value of lost load, or with price elasticity (linear demand function modelled as quadratic program). Cross-price elasticity between different hours is also possible. CES functions etc. are not possible.

### Can investments be modelled for different years over multiple decades?
Yes, cf. Lisa Zeyen’s paper. Slows things down of course.

### Can technological learning be modelled?
Yes with MILP piecewise-linear learning curve, cf. Lisa Zeyen's paper.

### Can you do Generation Adequacy Studies with Monte Carlo unplanned outages?
No, this functionality is not offered directly in PyPSA, but can be built in an outer loop around the PyPSA code.
TODO: Any OS code for this online?

### Does PyPSA have a GUI?
There is currently no desktop application where you can build a model from scratch without programming in Python. The usual mode of interaction with PyPSA is via Jupyter Notebooks, where you enter code to build and inspect the model and plot inputs and outputs. This is a form of GUI. There are also example [online scenario generators](https://model.energy/scenarios/), where you can enter inputs and start simulations for customised models.

### How easy is it to add custom constraints to PyPSA?
Easily in linopy framework on which PyPSA is based; you can inspect all constraints in PyPSA’s OS code

### Can you model market clearing in PyPSA?
Yes, if you take care with demand side, it’s just an optimisation problem.

### Does PyPSA do ancillary service co-optimization (e.g. frequency control)?
requires some minor customisation depending on what you want, but there are examples for reserves etc on the website

### Can you model intra-day as well as day-ahead markets?
Currently not without customisation.

### Is storage capacity optimized endogenously?
Yes.

### Does PyPSA offer stochastic optimisation?
Covered partially with multi-period optimisation, but truly stochastic LP scenarios are coming in next version 

### How does PyPSA model grid load flow physics?
DC-OPF, with the option to add piecewise linearised losses (link Neumann-Hagenmeyer-Brown paper)

### How are N-1 and line outages handled?
Security-constrained-DC-OPF is offered, can also simplify to 70% max loading (cf. Amin paper)

### Can transmission line projects be endogenously selected?
General mode is continuous expansion of offered lines, then post-discretisation, but you can also choose projects discretely with a MILP, see Fabian Neumann's first paper.

### How are capacity retirements and stranded assets handled?
Lisa knows.

### Is electric vehicle charging and V2G handled engodenously?
Yes.

### How do I model a concentrating solar plant?
See Johannes paper.

### How do I model retrofitting of coal plants with CCS?
See Xiaowei paper.

### How long does it take a PyPSA model to run?
dfgdf

### How much computing resource do I need to run a PyPSA model?
dfgdfg

### Do I need to buy a commercial solver?
No, but it helps for very large problems (100s of nodes, 1000s of time periods).

### How long does it take to learn PyPSA?
3-4 days for basics of Python, 1-2 days for pandas, 3-4 weeks for PyPSA basics.

### How long does it take to learn PyPSA-Eur?
1 month for basics, 2-3 months to start altering the model.

### Can I get support for using PyPSA?
Mailing list, discord forum, etc.

Also various consultancies offer paid support: OET, d-fine, energynautics, CLIMACT….

### Can you guarantee PyPSA will be developed further in the future?
We cannot guarantee anything, but there is a lively community of developers, several of whom have permanent contracts, and there is funding guaranteed until 2027 for a research engineer.

### Is there an automatic conversion from PLEXOS to PyPSA?
Not yet. PyPSA relies on data table inputs, so if you can convert to Excel or CSV, you can read into PyPSA.

### Is there an automatic conversion from Matpower to PyPSA?
There used to be, but it may be out of date. PyPSA relies on data table inputs, so if you can convert to Excel or CSV, you can read into PyPSA.