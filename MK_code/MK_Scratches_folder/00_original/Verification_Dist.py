import lettuce as lt
import torch
import numpy as np

def run(device, mpiObject):
    rank = mpiObject.rank
    size = mpiObject.size
    resolution = 500
    steps = 50000
    print(f"3D multi core test TGV, device: {device}, resolution: {resolution}, steps: {steps}")
    print(f"I am process {rank} of {size}")
    lattice = lt.Lattice(lt.D2Q9, device=device, dtype=torch.float32, MPIObject=mpiObject)
    #flow = lt.CouetteFlow2D(resolution, 10, 0.1, lattice)
    flow = lt.TaylorGreenVortex2D(resolution, 10, 0.1, lattice)
    #lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float32)
    #flow = lt.TaylorGreenVortex3D(resolution, 10, 0.1, lattice)
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow, lattice, collision, streaming)
    #rep = lt.VTKReporter(lattice, flow, steps-1, "./results/vtk")
    #simulation.reporters.append(rep)

    if rank == 0:
        lattice2 = lt.Lattice(lt.D2Q9, device=device, dtype=torch.float32)
        #flow2 = lt.CouetteFlow2D(resolution, 10, 0.1, lattice2)
        flow2 = lt.TaylorGreenVortex2D(resolution, 10, 0.1, lattice2)
        #lattice2 = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float32)
        #flow2 = lt.TaylorGreenVortex3D(resolution, 10, 0.1, lattice2)
        collision2 = lt.BGKCollision(lattice2, tau=flow2.units.relaxation_parameter_lu)
        streaming2 = lt.StandardStreaming(lattice2)
        simulation2 = lt.Simulation(flow2, lattice2, collision2, streaming2)
        #rep2 = lt.VTKReporter(lattice2, flow2, steps-1, "./results/vtk2")
        #simulation2.reporters.append(rep2)

    for i in range(0, steps):
        simulation.step(1)
        f = flow.rgrid.reassemble(simulation.f)

        if rank == 0:
            simulation2.step(1)
            #simulation2.f[5,5, 5] = simulation2.f[5,5,5] + 1*10**(-10)
            f2 = simulation2.f
            print("{:.20f}".format(torch.max(torch.abs(f-f2))))

if __name__ == "__main__":
    device = torch.device("cuda")
    #gpuList_siegen=[[4,"gpu-node001"],[4,"gpu-node002"],[4,"gpu-node003"],[4,"gpu-node004"],[1,"gpu-node005"],[1,"gpu-node006"],[1,"gpu-node007"],[1,"gpu-node008"],[2,"gpu-node009"],[2,"gpu-node010"]]
    gpuList_hbrs=[[4, "wr15"], [1, "wr12"], [1, "wr16"], [1, "wr17"], [1, "wr18"], [1, "wr19"]]
    mpiOBJ=lt.mpiObject(1, gpuList=gpuList_hbrs, setParts=0, gridRefinment=0, printUserInfo=1)
    lt.running(run, device, mpiOBJ)