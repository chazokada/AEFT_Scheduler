from os import system
from itertools import product

minalpha = 20
maxalpha = 50
n = [10, 20, 30, 40, 50]
fat = [0.1, 0.4, 0.8]
density = [0.2, 0.8]
regularity = [0.2, 0.8]
jump = [1,2,4]

keys = ['n', 'fat', 'density', 'regularity', 'jump']
values = [n, fat, density, regularity, jump]

for v in product(*values):
    param = dict(zip(keys, v))
    filename = 'dag/{}_{}_{}_{}_{}.dot'.format(param['n'],
        param['fat'],
        param['density'],
        param['regularity'],
        param['jump'])
    param = dict(zip(keys, v))
    system("daggen-master/daggen -n {} --fat {} --density {} --regular {} --jump {} --minalpha {} --maxalpha {} --dot -o {}".format(
        param['n'],
        param['fat'],
        param['density'],
        param['regularity'],
        param['jump'],
        minalpha,
        maxalpha,
        filename
    ))