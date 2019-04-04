import numpy as np
import healpy as hp
import os
from UFalcon import utils


def pos2ang(z_input, deltaz, path, boxsize, cosmo):
    """
    Iterates over all binary files within the desired folder and imports the particles within that file. The position of the particles are then stored in 4 array labeled by "a,b,c,d".
    The positions are then transformed to spherical coordinated (phi, theta) and projected onto different planes. The analysis is perfomed in comoving distances, i.g. in [Mpc*a]
    :return dic: defined dictionary dic with values corresponding to angular position on the planes, e.g. "theta_6Gpc_3_a"
    """

    dic = {}

    as_float = np.fromfile(path, dtype=np.float32, count=-1)
    # same data, but different interpretation of bit patterns
    as_uint = as_float.view(np.uint32)

    block_start = 0
    data_blocks = []

    while block_start < len(as_uint):

        #print("block header", as_uint[block_start: block_start + 4])
        block_size = as_uint[block_start + 1] * 7
        data_block = as_float[block_start + 4: block_start + 4 + block_size].reshape(-1, 7)
        data_blocks.append(data_block)

        #print("block has shape", data_block.shape)

        block_start = block_start + block_size + 5  # 4 uint32 before the block and 1 afterwards

    data = np.vstack(data_blocks)

    pos_x = data[:,0] / cosmo.params.h
    pos_y = data[:,1] / cosmo.params.h
    pos_z = data[:,2] / cosmo.params.h

    origin = boxsize * 500.0

    r = np.sqrt((pos_x - origin) ** 2 + (pos_y - origin) ** 2 + (pos_z - origin) ** 2)
    min_r, max_r = np.min(r), np.max(r)

    if (max_r < utils.comoving_distance(z_input, z_input + deltaz, cosmo)) or (min_r > utils.comoving_distance(z_input, z_input + deltaz, cosmo)):
            dic[1] = np.array([], np.float32)
            dic[2] = np.array([], np.float32)
    else:
        shell = np.where(np.logical_and(r > utils.comoving_distance(z_input, z_input + deltaz, cosmo), r <= utils.comoving_distance(z_input, z_input + deltaz, cosmo)))[0]
        shell_x = (pos_x - origin)[shell]
        shell_y = (pos_y - origin)[shell]
        shell_z = (pos_z - origin)[shell]

        dic[1] = np.pi / 2 - np.arctan2(shell_z, (np.sqrt(shell_x ** 2 + shell_y ** 2)))
        dic[2] = np.pi + np.arctan2(shell_y, shell_x)
        del shell_x, shell_y, shell_z

    del pos_x, pos_y, pos_z

    return dic

def n_part(dic, NSIDE):
    """
    Returns an array of length Npix with the number of particles per pixel.
    :param dic: dictionary containing x,y,z positions
    :param NSIDE: resolution of the healpy map
    :return: array of length Npix, whose elements correspond to the number of particles in that pixel
    """
    arr_theta = dic[1]
    arr_phi = dic[2]

    arr_theta = arr_theta[arr_theta != 0]
    arr_phi = arr_phi[arr_phi != 0]

    pix = hp.ang2pix(NSIDE, arr_theta, arr_phi, nest=False)
    pix_binned = np.bincount(pix)
    pixels = np.zeros(Npix)
    pixels[:len(pix_binned)] = pix_binned

    return pixels

def npart_map(z_input, deltaz, path):

    a = int(z_input)
    b = str(round(z_input - int(z_input), 2))[2:]

    if len(b) == 1:
        z_file = '{}p{}00'.format(a, b)
    else:
        z_file = '{}p{}0'.format(a, b)

    n_part_total = None

    for filename in os.listdir(path):
        filepath = os.path.join(z_input, deltaz, path, filename)

        if os.path.exists(filepath):
            print('extracting particles from: {}'.format(filepath))

            ang = pos2ang(z_input, filepath, cosmo)

            if n_part_total is None:
                n_part_total = n_part(ang, NSIDE)
            else:
                n_part_total += n_part(ang, NSIDE)
        else:
            pass

            print('shell construction is finished for z={}'.format(z_input))

    return n_part_total

