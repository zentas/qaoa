import pickle
import generate_random_matrix as grm
import experiment as exp

def save_models_to_file(filename, contents):
    with open(filename, 'wb') as f:
        pickle.dump(contents, f)

def read_models_from_file(filename):
    res = []
    try:
        with open(filename, 'rb') as f:
            while True:
                try:
                    res.extend(pickle.load(f))
                except EOFError:
                    break
    except FileNotFoundError:
        print("no models found, i will generate some more")
    return res


def generate_and_save_models_threshold(dim, dens, number):
    models_and_brute_solutions = generate_models_threshold(dim, dens, number)
    save_models_to_file(f'./saves/dim_{dim}_dens_{dens}.pkl', models_and_brute_solutions)

def generate_models_threshold(dim, dens, number):
    model_generator = exp.Experiment(p=2, dim=dim, method=None, create_backend=None)
    matrices = [grm.random_normal_matrix(dim, dens) for i in range(number)]
    return [(matrix, model_generator.brute_force(grm.model_from_matrix(matrix.A))) for matrix in matrices]

def read_models_threshold(dim, dens, number):
    matrices_and_brute_solutions = read_models_from_file(f'./saves/dim_{dim}_dens_{dens}.pkl')
    if (real_number := len(matrices_and_brute_solutions)) < number:
        print("not enough generated, generate new models")
        print("real_number - ", real_number, " number - ", number)
        additional_matrices = generate_models_threshold(dim, dens, number - real_number)
        save_models_to_file(f'./saves/dim_{dim}_dens_{dens}.pkl', additional_matrices)
        matrices_and_brute_solutions.extend(additional_matrices)
    return [(grm.model_from_matrix(m.A), b) for m, b in matrices_and_brute_solutions[:number]]


if __name__ == "__main__":
    my_dim = 3
    my_dens = 0.4
    my_number = 2

    print(read_models_threshold(3, 0.4, 3))







