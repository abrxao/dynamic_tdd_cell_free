from docplex.mp.model import Model
import numpy as np

qmdl= Model("quadratic")

x = qmdl.continuous_var(name="x", lb=1)
y = 2*x
z = qmdl.binary_var(name="z")

qmdl.add(z**2+y<=5)
obj_fn = x**2+y**2+z**2
qmdl.set_objective("min",obj_fn)
qmdl.print_information()
qmdl.solve()
qmdl.print_solution()