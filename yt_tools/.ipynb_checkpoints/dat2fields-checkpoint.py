import yt 
import aglio
import glob
import re
import os
import time
import logging
import pickle
import argparse
import json

import numpy as np 
from typing import List
from multiprocessing import Pool
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from yt_spherical_3D import *
# =========================================================================================== #

parser = argparse.ArgumentParser(description='convert AMRVAC .dat to your choosen fields file')
parser.add_argument('-f', type=str, help='folder path for the .dat files', required=False, default='./data/')
parser.add_argument('-n', type=str, help='fields you want to import', required=False, default='rho,e,m1,m2,m3,b1,b2,b3')
parser.add_argument('--config', type=str, help='configuration files path', required=False, default = None)

args        = parser.parse_args()
folder      = args.f
fields_list = args.n.split(',')
config      = args.config

# =================== initialization ========================= #
xyz_file         = './x_y_z.npz'
n_sph            = 400
n_car            = 400
fields_save_path = './yt_fields/'
is_reload        = False
reload_file      = None
save_pkl         = True
pkl_name         = 'spherical_data.pkl'
fields_file_name = 'fields'
fields_auto_skip = True
# =================== initialization ========================= #

if config != None:
    with open(config, 'r') as json_file:
        config_data = json.load(json_file)
        
    folder           = config_data['folder']
    fields_list      = config_data['fields_list']
    xyz_file         = config_data['xyz_file']
    n_sph            = config_data['n_sph']
    n_car            = config_data['n_car']
    fields_save_path = config_data['fields_save_path']
    is_reload        = config_data['is_reload']
    reload_file      = config_data['reload_file']
    save_pkl         = config_data['save_pkl']
    pkl_name         = config_data['pkl_name']
    fields_file_name = config_data['fields_file_name']
    fields_auto_skip = config_data['fields_auto_skip']
    
    
if not is_reload:
    tool = spherical_data(folder=folder, fields_list=fields_list, n_sph=n_sph, n_car=n_car, fields_save_path=fields_save_path)
else:
    tool = spherical_data.load(reload_file)
    
if save_pkl:
    tool.save(output_name=pkl_name)

tool.save_fields(fields_list=fields_list, fields_file_name=fields_file_name, auto_skip=fields_auto_skip)

print(' !!!!! Code End !!!!!')
