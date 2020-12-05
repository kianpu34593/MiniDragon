from mpi4py import MPI
import numba
import utils as ut

#@numba.njit()
def par_worker_vel_ver_pre(infodic,dt,r_limit,L):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    vel_Ver(infodic,dt,r_limit,L)
    
#@numba.njit()
def vel_Ver(infodict,dt,r_limit=2.5,L=6.8):
    comm = MPI.COMM_WORLD
    size = comm.Get_size() 
    rank = comm.Get_rank()
    if rank!=0:
      ## Get the data for the current spatial domain 
      my_spd, neighs_spd=ut.separate_points(infodict, rank)
      #acceleration at 0
      my_spd.A=ut.LJ_accel(position=my_spd.P,neighb_x_0=neighs_spd.P,r_cut=r_limit,L=L)
      #velocity at dt/2
      my_spd.V=my_spd.V+my_spd.A*(dt/2)
      #position at dt
      my_spd.P=my_spd.P+my_spd.V*dt
      #PBC rule 1
      my_spd.P=ut.pbc1(position=my_spd.P,L=L)
      #infodict[i]=my_spd
      #print('send',rank)
      comm.send(my_spd, dest=0)
    if rank == 0:
      for j in range (1,size):
        infodict[j]=comm.recv(source=j)
    infodict=comm.bcast(infodict,root=0)
    if rank!=0:   
      my_spd, neighs_spd=ut.separate_points(infodict, rank) 
      #acceleration at dt
      my_spd.A=ut.LJ_accel(position=my_spd.P,neighb_x_0=neighs_spd.P,r_cut=r_limit,L=L)
      #velocity at dt
      my_spd.V=my_spd.V+my_spd.A*(dt/2)
      comm.send(my_spd, dest=0)
    if rank == 0:
      for j in range (1,size):
        infodict[j]=comm.recv(source=j)
    return infodict 