"""
Input/output routines.
TODO: Logging
"""

import sys
import warnings
import os
import numpy as np
import torch
import pyevtk.hl as vtk
import datetime
import gc
from collections import Counter

__all__ = [
    "write_image", "write_vtk", "VTKReporter", "ObservableReporter", "ErrorReporter", "VRAMreporter"
]


def write_image(filename, array2d):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.imshow(array2d)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(filename)


def write_vtk(point_dict, id=0, filename_base="./data/output"):
    # vtk.gridToVTK(f"{filename_base}_{id:08d}",
    #               np.arange(0, point_dict["p"].shape[0]),
    #               np.arange(0, point_dict["p"].shape[1]),
    #               np.arange(0, point_dict["p"].shape[2]),
    #               pointData=point_dict)

    vtk.imageToVTK(
        path=f"{filename_base}_{id:08d}",
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        cellData=None,
        pointData=point_dict,
        fieldData=None,
    )


class VTKReporter:
    """General VTK Reporter for velocity and pressure"""

    def __init__(self, lattice, flow, interval=50, filename_base="./data/output"):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.filename_base = filename_base
        directory = os.path.dirname(filename_base)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        self.point_dict = dict()

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))
            #rho = self.flow.units.convert_density_to_pu(self.lattice.rho(f))
            if self.lattice.D == 2:
                self.point_dict["p"] = self.lattice.convert_to_numpy(p[0, ..., None])
                for d in range(self.lattice.D):
                    self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ..., None])
                #self.point_dict["rho"] = self.lattice.convert_to_numpy(rho[0, ..., None])
            else:
                self.point_dict["p"] = self.lattice.convert_to_numpy(p[0, ...])
                for d in range(self.lattice.D):
                    self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ...])
                #self.point_dict["rho"] = self.lattice.convert_to_numpy(rho[0, ...])
            write_vtk(self.point_dict, i, self.filename_base)

    def output_mask(self, no_collision_mask):
        """Outputs the no_collision_mask of the simulation object as VTK-file with range [0,1]
        Usage: vtk_reporter.output_mask(simulation.no_collision_mask)"""
        point_dict = dict()
        if self.lattice.D == 2:
            point_dict["mask"] = self.lattice.convert_to_numpy(no_collision_mask)[..., None].astype(int)
        else:
            point_dict["mask"] = self.lattice.convert_to_numpy(no_collision_mask).astype(int)
        vtk.gridToVTK(self.filename_base + "_mask",
                      np.arange(0, point_dict["mask"].shape[0]),
                      np.arange(0, point_dict["mask"].shape[1]),
                      np.arange(0, point_dict["mask"].shape[2]),
                      pointData=point_dict)


class ErrorReporter:
    """Reports numerical errors with respect to analytic solution."""

    def __init__(self, lattice, flow, interval=1, out=sys.stdout):
        assert hasattr(flow, "analytic_solution")
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.out = [] if out is None else out
        if not isinstance(self.out, list):
            print("#error_u         error_p", file=self.out)

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            pref, uref = self.flow.analytic_solution(self.flow.grid, t=t)
            pref = self.lattice.convert_to_tensor(pref)
            uref = self.lattice.convert_to_tensor(uref)
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))

            resolution = torch.pow(torch.prod(self.lattice.convert_to_tensor(p.size())), 1 / self.lattice.D)

            err_u = torch.norm(u - uref) / resolution ** (self.lattice.D / 2)
            err_p = torch.norm(p - pref) / resolution ** (self.lattice.D / 2)

            if isinstance(self.out, list):
                self.out.append([err_u.item(), err_p.item()])
            else:
                print(err_u.item(), err_p.item(), file=self.out)


class ObservableReporter:
    """A reporter that prints an observable every few iterations.

    Examples
    --------
    Create an Enstrophy reporter.

    >>> from lettuce import TaylorGreenVortex3D, Enstrophy, D3Q27, Lattice
    >>> lattice = Lattice(D3Q27, device="cpu")
    >>> flow = TaylorGreenVortex(50, 300, 0.1, lattice)
    >>> enstrophy = Enstrophy(lattice, flow)
    >>> reporter = ObservableReporter(enstrophy, interval=10)
    >>> # simulation = ...
    >>> # simulation.reporters.append(reporter)
    """

    def __init__(self, observable, interval=1, out=sys.stdout):
        self.observable = observable
        self.interval = interval
        self.out = [] if out is None else out
        self._parameter_name = observable.__class__.__name__
        if out is not None:
            print('steps    ', 'time    ', self._parameter_name)

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            observed = self.observable.lattice.convert_to_numpy(self.observable(f))
            assert len(observed.shape) < 2
            if len(observed.shape) == 0:
                observed = [observed.item()]
            else:
                observed = observed.tolist()
            entry = [i, t] + observed
            if isinstance(self.out, list):
                self.out.append(entry)
            else:
                print(*entry, file=self.out)

class VRAMreporter:

    def __init__(self, interval=1000, filename_base="./vram_data/vram_summary"):
        self.interval = interval
        self.filename_base = filename_base
        directory = os.path.dirname(filename_base)
        if not os.path.isdir(directory):
            os.mkdir(directory)
    def __call__(self, i, t, f):
        if i % self.interval == 0:
            # export CUDA-VRAM-summary + index
            # get tensors, export under generic filename
            # export tensor-count
            # count tensors and export tensor-count with byte-summary + index
            timestamp = datetime.datetime.now()
            timestamp = timestamp.strftime("%y%m%d") + "_" + timestamp.strftime("%H%M%S")
            output_file = open(self.filename_base + "_" + timestamp + "_GPU_VRAM_summary_"+str(i)+".txt", "a")
            output_file.write("DATA for " + timestamp + "\n\n")
            output_file.write(torch.cuda.memory_summary(device="cuda:0"))
            output_file.close()

            ### list present torch tensors:
            output_file = open(self.filename_base + "_temp_GPU_list_of_tensors.txt", "a")
            total_bytes = 0
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        output_file.write("\n" + str(obj.size()) + ", " + str(obj.nelement() * obj.element_size()))
                        total_bytes = total_bytes + obj.nelement() * obj.element_size()
                except:
                    pass
            #output_file.write("\n\ntotal bytes for tensors:" + str(total_bytes))
            output_file.close()

            ### count occurence of tensors in list of tensors:
            my_file = open(self.filename_base + "_temp_GPU_list_of_tensors.txt", "r")
            data = my_file.read()
            my_file.close()
            data_into_list = data.split("\n")
            c = Counter(data_into_list)
            output_file = open(self.filename_base + "_" + timestamp + "_GPU_counted_tensors_"+str(i)+".txt", "a")
            for k, v in c.items():
                output_file.write("type,size,bytes: {}, number: {}\n".format(k, v))
            output_file.write("\ntotal bytes for tensors:" + str(total_bytes))
            output_file.close()

            if os.path.exists(self.filename_base + "_temp_GPU_list_of_tensors.txt"):
                os.remove(self.filename_base + "_temp_GPU_list_of_tensors.txt")