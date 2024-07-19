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
from timeit import default_timer as timer
import gc
from collections import Counter

__all__ = [
    "write_image", "write_vtk", "VTKReporter", "ObservableReporter", "ErrorReporter",
    "VRAMreporter", "Clock", "NaNReporter", "AverageVelocityReporter", "Watchdog"
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
    # OLD Master-version
    # vtk.gridToVTK(f"{filename_base}_{id:08d}",
    #               np.arange(0, point_dict["p"].shape[0]),
    #               np.arange(0, point_dict["p"].shape[1]),
    #               np.arange(0, point_dict["p"].shape[2]),
    #               pointData=point_dict)

    # version by M.C.B.
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
    "EDIT (M.Bille: can insert solid-mask to pin osb. to zero, inside solid obstacle, " \
    "...useful for boundaries that store populations inside the boundary-region (FWBB, HWBBc3,...), making obs(f) deviate from 0"

    def __init__(self, lattice, flow, interval=50, filename_base="./data/output", solid_mask=None, imin=0):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.filename_base = filename_base
        self.imin=imin
        if solid_mask is not None and lattice.D == 2:
            self.solid_mask = solid_mask[..., None]
        else:
            self.solid_mask = solid_mask
        directory = os.path.dirname(filename_base)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        self.point_dict = dict()

    def __call__(self, i, t, f):
        if i % self.interval == 0 and i >= self.imin:
            u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(f))
            #rho = self.flow.units.convert_density_to_pu(self.lattice.rho(f))
            if self.lattice.D == 2:
                if self.solid_mask is None:
                    self.point_dict["p"] = self.lattice.convert_to_numpy(p[0, ..., None])
                else:
                    self.point_dict["p"] = np.where(self.solid_mask, 0, self.lattice.convert_to_numpy(p[0, ..., None]))
#ALTERNATIVE:                self.point_dict["p"] = self.lattice.convert_to_numpy(torch.where(self.solid_mask, 0, p[0, ..., None]))  # for boundaries that store populations "inside" the boundary
                for d in range(self.lattice.D):
                    if self.solid_mask is None:
                        self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ..., None])
                    else:
                        self.point_dict[f"u{'xyz'[d]}"] = np.where(self.solid_mask, 0, self.lattice.convert_to_numpy(u[d, ..., None]))
#ALTERNATIVE:                    self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(torch.where(self.solid_mask, 0, u[d, ..., None]))
                #self.point_dict["rho"] = self.lattice.convert_to_numpy(rho[0, ..., None])
            else:
                if self.solid_mask is None:
                    self.point_dict["p"] = self.lattice.convert_to_numpy(p[0, ...])
                else:
                    self.point_dict["p"] = np.where(self.solid_mask, 0, self.lattice.convert_to_numpy(p[0, ...]))
                #ORIGINAL: self.point_dict["p"] = self.lattice.convert_to_numpy(p[0, ...])
#ALTERNATIVE:               self.point_dict["p"] = self.lattice.convert_to_numpy(torch.where(self.solid_mask, 0, p[0, ...]))
                for d in range(self.lattice.D):
                    #ORIGINAL: self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ...])
                    if self.solid_mask is None:
                        self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ...])
                    else:
                        self.point_dict[f"u{'xyz'[d]}"] = np.where(self.solid_mask, 0, self.lattice.convert_to_numpy(u[d, ...]))

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


class Clock:
    """reports t_LU (step) and t_PU"""
    def __init__(self, lattice, interval=1, start=1):
        self.lattice = lattice
        self.interval = interval
        self.start = start
        self.out = []

    def __call__(self, i, t, f):
        if i % self.interval == 0 and i >= self.start:
            self.out.append([i, t])


class NaNReporter:
    """reports any NaN and aborts the simulation"""
    def __init__(self, flow, lattice, n_target, t_target, interval=100, ):
        self.flow = flow
        self.lattice = lattice
        self.interval = interval
        self.n_target = n_target
        self.t_target = t_target

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            if torch.isnan(f).any() == True:
                if self.lattice.D == 2:
                    q, x, y = torch.where(torch.isnan(f))
                    q = self.lattice.convert_to_numpy(q)
                    x = self.lattice.convert_to_numpy(x)
                    y = self.lattice.convert_to_numpy(y)
                    nan_location = np.stack((q, x, y), axis=-1)
                    print("(!) NaN detected at (q,x,y):", nan_location)
                if self.lattice.D == 3:
                    q, x, y, z = torch.where(torch.isnan(f))
                    q = self.lattice.convert_to_numpy(q)
                    x = self.lattice.convert_to_numpy(x)
                    y = self.lattice.convert_to_numpy(y)
                    z = self.lattice.convert_to_numpy(z)
                    nan_location = np.stack((q, x, y, z), axis=-1)
                    print("(!) NaN detected at (q,x,y,z):", nan_location)
                print("(!) NaN detected in time step", i, "of", self.n_target, "(interval:", self.interval, ")")
                print("(!) Aborting simulation at t_PU", self.flow.units.convert_time_to_pu(i), "of", self.t_target)
                sys.exit()


class AverageVelocityReporter:
    """Reports the streamwise velocity averaged in span direction (z) at defined position in x.
        Refer to Di Ilio et al. 2018 for further references
        out = averaged u profile at position
        t_out = t_LU and t_PU
        """
    def __init__(self, lattice, flow, position, interval=1, start=1):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.start = start  # step to start with reporting
        self.t_out = []
        self.out = []
        self.x_position = int(round(position, 0))  # rounded LU position

        # linear interpolation of u for x_pos off grid
        if position % 1 != 0:
            self.interpol = True
            self.x_pos1 = int(np.floor(position))
            self.x_pos2 = int(np.ceil(position))
            self.w1 = position - self.x_pos1
            self.w2 = 1 - self.w1

    def __call__(self, i, t, f):
        if i % self.interval == 0 and i >= self.start:
            if self.interpol:
                u = self.lattice.u(f)[:, self.x_pos1] * self.w1 + self.lattice.u(f)[:, self.x_pos2] * self.w2
            else:
                u = self.lattice.u(f)[:, self.x_position]
            u = self.flow.units.convert_velocity_to_pu(u).cpu().numpy()
            self.t_out.append([i, t])
            if self.lattice.D == 2:
                self.out.append(u)
            elif self.lattice.D == 3:
                self.out.append(np.mean(u, axis=2))


def append_txt_file(filename, line):
    file = open(filename, "a")
    file.write(line + "\n")
    file.close()

class Watchdog:
    '''
        Watchdog reporter that logs time, elapsed time, elapsed steps and estimates time remaining.
        Additionally writes a checkpoint file, if t_max is reached.

        Can be used to estimate time remaining and to get a checkpoint file if sim would run longer than allowed by the host.
        Sim can be restarted from checkpoint.
        (!) Watchdog reporter does not export other reporters observable values etc., so make sure you save them in other ways, if sim is stopped by host system!

    '''

    def __init__(self, lattice, flow, sim, interval=1000, i_start=0, i_target=1, t_max=71*3600, filebase="./watchdog"):
        self.interval = interval
        self.lattice = lattice
        self.flow = flow
        self.sim = sim
        self.i_start = i_start
        self.i_target = i_target
        self.t_max = t_max
        self.filebase = filebase
        try:
            os.makedirs(filebase)
        except FileExistsError:
            # directory already exists
            pass
        self.running = False
        self.t_start = 0

    def __call__(self, i, t, f):
        #print("calling watchdog")
        if not self.running:
            self.start_timer()
        elif i % self.interval == 0:
            #print("watchdog in interval")
            #print("watchdog with", str(i))
            timestamp = datetime.datetime.now()
            timestamp_str = timestamp.strftime("%y%m%d") + "_" + timestamp.strftime("%H%M%S")

            t_now = timer()
            t_elapsed = t_now - self.t_start
            t_per_step = t_elapsed/(i-self.i_start)
            i_remaining = self.i_target - i
            t_remaining_estimate = t_per_step * i_remaining
            datetime_finish_estimate = timestamp + datetime.timedelta(seconds=t_remaining_estimate)
            t_total_estimate = t_elapsed + t_remaining_estimate

            # write DATA and warn if t_total_estimate > t_max
            if t_total_estimate > self.t_max:
                append_txt_file(self.filebase+"/log.txt", timestamp_str.ljust(13) + " " + str(i).rjust(10) + " " + "{:.2f}".format(t_now).rjust(10) + " " + "{:.2f}".format(t_elapsed).rjust(10) + " " + "{:.6f}".format(t_per_step).rjust(10) + " " + "{:.2f}".format(t_remaining_estimate).rjust(15) + " " + "{:.2f}".format(t_total_estimate).rjust(15) + " " + str(datetime_finish_estimate).ljust(26) + " WARNING t_total>t_max=" + str(self.t_max))
            else:
                append_txt_file(self.filebase+"/log.txt", timestamp_str.ljust(13) + " " + str(i).rjust(10) + " " + "{:.2f}".format(t_now).rjust(10) + " " + "{:.2f}".format(t_elapsed).rjust(10) + " " + "{:.6f}".format(t_per_step).rjust(10) + " " + "{:.2f}".format(t_remaining_estimate).rjust(15) + " " + "{:.2f}".format(t_total_estimate).rjust(15) + " " + str(datetime_finish_estimate).ljust(26))
            # write checkpoint if t_elapsed > t_max
            if t_elapsed > self.t_max:
                self.sim.save_checkpoint(self.filebase+"/"+timestamp_str + "_f_"+str(self.sim.i)+".cpt")

    def start_timer(self):
        self.running = True
        self.t_start = timer()
        #print("starting timer")
        append_txt_file(self.filebase+"/log.txt", "t_start: "+str(self.t_start)+", interval: "+str(self.interval)+", i_target: "+str(self.i_target))
        append_txt_file(self.filebase+"/log.txt", "timestamp ".center(13)+"|"+"step".center(10)+"|"+"t_now".center(10)+"|"+"t_elapsed".center(10)+"|"+"t_per_step".center(10)+"|"+"t_remain(est)".center(15)+"|"+"t_total(est)".center(15)+"|"+"DATE_FINISH(est)".center(26)+"|"+" T WARNING")
        # sizes:                                   13, 10, 7+2.(rjust10), 7+2.(rjust10), 1+6.(rjust10), 7+2.(rjust10), 7+2.(rjust15), 27
