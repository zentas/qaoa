from pyqubo import Constraint, Array

def model_from_graph(graph, h):
    nodes = graph.number_of_nodes()

    x = Array.create('x', shape=(nodes, nodes), vartype='BINARY')
    c = Array.create('c', shape=(nodes), vartype='BINARY')  

    constr1 = Constraint(x[0, 0] * 0, "")
    for i in range(nodes):
        for k in range(nodes):
            for l in range(nodes):
                if l != k:
                    constr1 += Constraint(x[i, k] * x[i, l], "")

    constr2 = Constraint(x[0, 0] * 0, "")
    for i in range(nodes):
        for k in range(nodes):
            constr2 += Constraint(1 - x[i, k], "")

    constr3 = Constraint(x[0, 0] * 0, "")
    for i, l in graph.edges:
        for k in range(nodes):
            constr3 += 2 * x[i, k] * x[l, k]

    constr4 = Constraint(x[0, 0] * 0, "")
    for i in range(nodes):
        for k in range(nodes):
            constr4 += Constraint(x[i, k] * (1 - c[k]), "")

    constr5 = Constraint(x[0, 0] * 0, "")
    for k in range(nodes):
        constr5 += c[k]

    alpha = nodes + h
    beta = (nodes**3 + nodes**2 + 1)*alpha + h

    return (beta * constr1 + alpha * constr2 + alpha * constr3 + alpha * constr4 + constr5).compile()
