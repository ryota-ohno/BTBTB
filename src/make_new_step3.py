##pbepbe+d3bjで計算
import os
import numpy as np
import pandas as pd
import subprocess
from utils import Rod, R2atom

############################汎用関数###########################
def get_monomer_xyzR(monomer_name,Ta,Tb,Tc,A2,A3):
    T_vec = np.array([Ta,Tb,Tc])
    df_mono=pd.read_csv(f'/home/ohno/Working/BTBTB/monomer/{monomer_name}.csv')
    atoms_array_xyzR=df_mono[['X','Y','Z','atom']].values
    xyz_array = atoms_array_xyzR[:,:3];R_array = atoms_array_xyzR[:,3].reshape((-1,1))

    ex = np.array([1.,0.,0.]); ez = np.array([0.,0.,1.])
    xyz_array = np.matmul(xyz_array,Rod(-ex,A2).T)#
    xyz_array = np.matmul(xyz_array,Rod(ez,A3).T)#
    xyz_array = xyz_array + T_vec
    
    return np.concatenate([xyz_array,R_array],axis=1)
        
def get_xyzR_lines(xyzR_array,file_description):
    lines = [     
        '%mem=40GB\n',
        '%nproc=40\n',
        '#P TEST pbepbe/6-311G** EmpiricalDispersion=GD3BJ counterpoise=2\n',
        '\n',
        file_description+'\n',
        '\n',
        '0 1 0 1 0 1\n'
    ]
    mol_len = len(xyzR_array)//2
    atom_index = 0
    mol_index = 0
    for x,y,z,atom in xyzR_array:
        mol_index = atom_index//mol_len + 1
        line = '{}(Fragment={}) {} {} {}\n'.format(atom,mol_index,x,y,z)     
        lines.append(line)
        atom_index += 1
    return lines

# 実行ファイル作成
def get_one_exe(file_name,machine_type):
    file_basename = os.path.splitext(file_name)[0]
    if machine_type==1:
        group=1;core=40
    elif machine_type==2:
        group=2;core=52    
    cc_list=[
        '#!/bin/sh \n',
         '#$ -S /bin/sh \n',
         '#$ -cwd \n',
         '#$ -V \n',
         f'#$ -q gr{group}.q \n',
         f'#$ -pe OpenMP {core} \n',
         '\n',
         'hostname \n',
         '\n',
         'export g16root=/home/g03 \n',
         'source $g16root/g16/bsd/g16.profile \n',
         '\n',
         'export GAUSS_SCRDIR=/home/scr/$JOB_ID \n',
         'mkdir /home/scr/$JOB_ID \n',
         '\n',
         'g16 < {}.inp > {}.log \n'.format(file_basename,file_basename),
         '\n',
         'rm -rf /home/scr/$JOB_ID \n',
         '\n',
         '\n',
         '#sleep 5 \n']
#          '#sleep 500 \n'

    return cc_list

######################################## 特化関数 ########################################

##################gaussview##################
def make_gaussview_xyz(auto_dir,monomer_name,params_dict,isInterlayer=False):
    a_ = params_dict['a']; b_ = params_dict['b']
    z = params_dict['z']; A2 = params_dict['A2']; A3 = params_dict['theta']
    cx = params_dict['cx']; cy = params_dict['cy']; cz = params_dict['cz']

    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A2,A3)
    monomer_array_b1 = get_monomer_xyzR(monomer_name,0,b_,2*z,A2,A3)
    monomer_array_a1 = get_monomer_xyzR(monomer_name,a_,0,0,A2,A3)
    monomer_array_b2 = get_monomer_xyzR(monomer_name,0,-b_,-2*z,A2,A3)
    monomer_array_a2 = get_monomer_xyzR(monomer_name,-a_,0,0,A2,A3)
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a_/2,b_/2,z,A2,-A3)
    monomer_array_t2 = get_monomer_xyzR(monomer_name,a_/2,-b_/2,-z,A2,-A3)
    monomer_array_t3 = get_monomer_xyzR(monomer_name,-a_/2,-b_/2,-z,A2,-A3)
    monomer_array_t4 = get_monomer_xyzR(monomer_name,-a_/2,b_/2,z,A2,-A3)
    monomer_array_c = get_monomer_xyzR(monomer_name,cx,cy,cz,A2,A3)
    a =np.array([a_,0,0])
    b =np.array([0,b_,0])

    monomers_array = np.concatenate([monomer_array_c,monomer_array_i,monomer_array_a1,monomer_array_a2,monomer_array_b1,monomer_array_b2,monomer_array_t1,monomer_array_t2,monomer_array_t3,monomer_array_t4],axis=0)
    
    file_description = '_z={}_A2={}_A3={}_cx={}_cx={}_cx={}'.format(np.round(z,1),round(A2),round(A3),cx,cy,cz)
    lines = get_xyzR_lines(monomers_array,file_description)
    lines.append('Tv {} {} {}\n'.format(a[0],a[1],a[2]))
    lines.append('Tv {} {} {}\n'.format(b[0],b[1],b[2]))
    #lines.append('Tv {} {} {}\n\n\n'.format(c[0],c[1],c[2]))
    
    os.makedirs(os.path.join(auto_dir,'gaussview'),exist_ok=True)
    output_path = os.path.join(
        auto_dir,
        'gaussview/{}_z={}_A2={}_A3={}_a={}_b={}_cx={}_cx={}_cx={}.gjf'.format(monomer_name,round(z),round(A2),round(A3),np.round(a_,2),np.round(b_,2),cx,cy,cz)
    )
            
    with open(output_path,'w') as f:
        f.writelines(lines)

def make_gjf_xyz(auto_dir,monomer_name,params_dict,isInterlayer):
    a_ = params_dict['a']; b_ = params_dict['b']
    z = params_dict['z']; A2 = params_dict['A2']; A3 = params_dict['theta']
    cx = params_dict['cx']; cy = params_dict['cy']; cz = params_dict['cz']

    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A2,A3)
    monomer_array_b1 = get_monomer_xyzR(monomer_name,0,b_,2*z,A2,A3)
    monomer_array_a1 = get_monomer_xyzR(monomer_name,a_,0,0,A2,A3)
    monomer_array_b2 = get_monomer_xyzR(monomer_name,0,-b_,-2*z,A2,A3)
    monomer_array_a2 = get_monomer_xyzR(monomer_name,-a_,0,0,A2,A3)
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a_/2,b_/2,z,A2,-A3)
    monomer_array_t2 = get_monomer_xyzR(monomer_name,a_/2,-b_/2,-z,A2,-A3)
    monomer_array_c = get_monomer_xyzR(monomer_name,cx,cy,cz,A2,A3)

    dimer_array_i = np.concatenate([monomer_array_c,monomer_array_i])
    dimer_array_t1 = np.concatenate([monomer_array_c,monomer_array_t1])
    dimer_array_t2 = np.concatenate([monomer_array_c,monomer_array_t2])
    dimer_array_a1 = np.concatenate([monomer_array_c,monomer_array_a1])
    dimer_array_a2 = np.concatenate([monomer_array_c,monomer_array_a2])
    dimer_array_b1 = np.concatenate([monomer_array_c,monomer_array_b1])
    dimer_array_b2 = np.concatenate([monomer_array_c,monomer_array_b2])
    
    file_description = '{}_z={}_A2={}_A3={}'.format(monomer_name,round(z,1),int(A2),round(A3,2))
    line_list_dimer_i = get_xyzR_lines(dimer_array_i,file_description+'_i')
    line_list_dimer_a1 = get_xyzR_lines(dimer_array_a1,file_description+'_a1')
    line_list_dimer_a2 = get_xyzR_lines(dimer_array_a2,file_description+'_a2')
    line_list_dimer_b1 = get_xyzR_lines(dimer_array_b1,file_description+'_b1')
    line_list_dimer_b2 = get_xyzR_lines(dimer_array_b2,file_description+'_b2')
    line_list_dimer_t1 = get_xyzR_lines(dimer_array_t1,file_description+'_t1')
    line_list_dimer_t2 = get_xyzR_lines(dimer_array_t2,file_description+'_t2')
    
    gij_xyz_lines = ['$ RunGauss\n'] + line_list_dimer_i + ['\n\n--Link1--\n'] + line_list_dimer_t1 + ['\n\n--Link1--\n'] + line_list_dimer_t2 + ['\n\n--Link1--\n'] + line_list_dimer_a1 + ['\n\n--Link1--\n'] + line_list_dimer_a2 + ['\n\n--Link1--\n'] + line_list_dimer_b1 + ['\n\n--Link1--\n'] + line_list_dimer_b2 + ['\n\n\n']#+ ['\n\n--Link1--\n'] + line_list_dimer_p2 + ['\n\n\n']
    
    file_name = get_file_name_from_dict(monomer_name,params_dict)
    inp_dir = os.path.join(auto_dir,'gaussian')
    gij_xyz_path = os.path.join(inp_dir,file_name)
    with open(gij_xyz_path,'w') as f:
        f.writelines(gij_xyz_lines)
    
    return file_name

def get_file_name_from_dict(monomer_name,paras_dict):
    file_name = ''
    file_name += monomer_name
    for key,val in paras_dict.items():
        val = np.round(val,2)
        file_name += '_{}={}'.format(key,val)
    return file_name + '.inp'
    
def exec_gjf(auto_dir, monomer_name, params_dict,machine_type,isInterlayer,isTest=True):
    inp_dir = os.path.join(auto_dir,'gaussian')
    print(params_dict)
    
    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict,isInterlayer)
    cc_list = get_one_exe(file_name,machine_type)
    sh_filename = os.path.splitext(file_name)[0]+'.r1'
    sh_path = os.path.join(inp_dir,sh_filename)
    with open(sh_path,'w') as f:
        f.writelines(cc_list)
    if not(isTest):
        subprocess.run(['qsub',sh_path])
    log_file_name = os.path.splitext(file_name)[0]+'.log'
    return log_file_name
    
############################################################################################