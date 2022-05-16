# NOTE: MULTIPROCESSING DOESNT WORK IN IPYNB FILES

from multiprocessing import Process
from multiprocessing import Pool

def square(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(square, [1, 2, 3]))


def print_name(name, name2):
    print('hello', name, 'and', name2)

if __name__ == '__main__':
    p = Process(target=print_name, args=('bob','alice'))
    p.start()
    p.join()


