from mpi4py import MPI
import numpy as np
import utils as ut
import utils_parallel as utp
def LJ_MD(subdiv,position_init,dt,stop_step,accel_init,r_cut,L,T_eq,e_scale,sig):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #initialization
    if rank == 0:
      k_B=1.38064852*10**(-23)
      size_sim=position_init.shape[0]
      x_dot_init=ut.random_vel_generator(size_sim,T_eq,e_scale)
    #initialize the info matrix
      info=np.zeros((stop_step+1,size_sim,9))
      info[0,:,:]=np.concatenate((position_init,x_dot_init,accel_init),axis=1)
      infotodic=ut.cell_to_obj((info[0,:,:]),subdiv[0],subdiv[1],subdiv[2],L)
    #initialize PE, KE, T_insta, P_insta, Momentum
      PE=np.zeros((stop_step+1,1))
      KE=np.zeros((stop_step+1,1))
      T_insta=np.zeros((stop_step+1,1))
      P_insta=np.zeros((stop_step+1,1))
    #zero step value
      PE[0,:]=ut.LJ_potent_nondimen(info[0,:,0:3],r_cut=r_cut,L=L)
      KE[0,:]=ut.Kin_Eng(info[0,:,3:6])
      T_insta[0,:]=2*KE[0,:]*e_scale/(3*(size_sim-1)*k_B) #k
      P_insta[0,:]=ut.insta_pressure(L,T_insta[0],info[0,:,0:3],r_cut,e_scale) #unitless
    else:
      infotodic=None
    comm.barrier()
    infotodic=comm.bcast(infotodic,root=0)
    for step in range(stop_step):
        if rank!=0:
          #call the vel_verlet parallel function
          utp.par_worker_vel_ver_pre(infotodic,dt,r_cut,L)
        else:
          info_temp=utp.vel_Ver(infodict=infotodic,dt=dt,r_limit=r_cut,L=L)
          tmp=ut.concatDict(info_temp)
          info[step+1,:,:]=np.concatenate((tmp.P,tmp.V,tmp.A),1)
          #UPDATE CUBES MAKE SURE ATOMS ARE IN RIGHT CUBES
          infotodic=ut.cell_to_obj(info[step+1,:,:],subdiv[0],subdiv[1],subdiv[2],L)
          #calculate and store PE, KE, T_insta, P_insta
          PE[step+1,:]=ut.LJ_potent_nondimen(info[step+1,:,0:3],r_cut=r_cut,L=L)
          KE[step+1,:]=ut.Kin_Eng(info[step+1,:,3:6])
          T_insta[step+1,:]=2*KE[step+1,:]*e_scale/(3*(size_sim-1)*k_B) #k
          P_insta[step+1,:]=ut.insta_pressure(L,T_insta[step+1],info[step+1,:,0:3],r_cut,e_scale) #unitless
        comm.barrier()
        infotodic=comm.bcast(infotodic,root=0)
    if rank==0:
      return info,PE,KE,T_insta,P_insta
