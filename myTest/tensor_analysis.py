from copy import deepcopy
import numpy as np
import lettuce as lt
import numpy as np
from lettuce.flows.obstacleCylinder import ObstacleCylinder
import torch
import time

from collections import Counter
import gc

counter = 1

# copy contents of this method whereever you want to make a cuda-memory-summary
def tensor_report_combined(identifier="noID"):
    output_file = open("/home/mbille/Desktop/test_tensors/VRAM_summary.txt", "a")
    output_file.write(str(identifier) +", current VRAM: " + str(torch.cuda.memory_allocated(device="cuda:0"))+", peak VRAM: " + str(torch.cuda.max_memory_allocated(device="cuda:0"))+"\n")
    output_file.close()

def tensor_report(counter):
    output_file = open("/home/mbille/Desktop/test_tensors/GPU_memory_summary_" + str(counter) + ".txt", "a")
    output_file.write(torch.cuda.memory_summary(device="cuda:0"))
    output_file.close()
    print("current VRAM (byte): counter, value", counter, torch.cuda.memory_allocated(device="cuda:0"))
    print("max. VRAM (byte): counter, value", counter, torch.cuda.max_memory_allocated(device="cuda:0"))

    ### list present torch tensors:
    output_file = open("/home/mbille/Desktop/test_tensors/_GPU_list_of_tensors.txt", "w")
    total_bytes = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                output_file.write("\n" + str(obj.size()) + ", " + str(obj.nelement() * obj.element_size()))
                total_bytes = total_bytes + obj.nelement() * obj.element_size()
        except:
            pass
    output_file.close()
    print("total bytes for tensors:" + str(total_bytes) + "; counter: " + str(counter))

    ### count occurence of tensors in list of tensors:
    my_file = open("/home/mbille/Desktop/test_tensors/_GPU_list_of_tensors.txt", "r")
    data = my_file.read()
    my_file.close()
    data_into_list = data.split("\n")
    c = Counter(data_into_list)
    output_file = open("/home/mbille/Desktop/test_tensors/_GPU_counted_tensors_" + str(counter) + ".txt", "a")
    for k, v in c.items():
        output_file.write("type,size,bytes: {}, number: {}\n".format(k, v))
    output_file.write("\n\ntotal bytes for tensors:" + str(total_bytes))
    output_file.close()

    return counter + 1

# counter = tensor_report(counter)  # lässt sich nicht ausführen

lattice = lt.Lattice(lt.D3Q27, "cuda:0", dtype=torch.float64)

counter = tensor_report(counter)

flow = ObstacleCylinder(shape=(2000,
                               1000,1),
                        reynolds_number=200, mach_number=0.01,
                        lattice=lattice,
                        char_length_pu=1,
                        char_length_lu=5,
                        char_velocity_pu=1,
                        lateral_walls='periodic',
                        bc_type='ibb1',
                        perturb_init=True,
                        u_init=0
                        )

counter = tensor_report(counter)

#boundarylist = deepcopy(flow.boundaries)
tau = flow.units.relaxation_parameter_lu
sim = lt.Simulation(flow, lattice,lt.BGKCollision(lattice, tau),lt.StandardStreaming(lattice))

counter = tensor_report(counter)

print(sim.step(1))

counter = tensor_report(counter)
pass