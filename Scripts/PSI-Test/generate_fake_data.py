import random
import math


def generate_random_ids_to_file(m,  filename):
    ids = random.sample(range(0, int(m*1.2)), m)
    with open(filename, 'w') as file:
        for i in range(m):
            file.write(str(ids[i]) + '\n')


def generate_random_numbers_to_file(m, n, filename):
    with open(filename, 'w') as file:
        for _ in range(m):
            random_numbers = [str(random.randint(1, 10000)) for _ in range(n)]
            # Join the random numbers with spaces and write them to the file
            line = " ".join(random_numbers)
            file.write(line + '\n')


# Specify the number of rows and the number of random numbers per row, as well as the filename to write to
m = 20  # For example, generate 5mrows
n = 7  # n random numbers per row
base_path = "./Player-Data/PSI/"
pn = 2

for i in range(pn):
    generate_random_ids_to_file(m, base_path+'ID_P'+str(i))
    generate_random_numbers_to_file(m, n, base_path+'F_P'+str(i))