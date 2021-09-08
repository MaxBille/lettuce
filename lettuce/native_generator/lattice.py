from . import *


class NativeLattice:
    def generate_rho(self, generator: 'GeneratorKernel'):
        if not generator.registered('rho'):
            generator.register('rho')

            # generate
            f_eq_sum = ' + '.join([f"f_reg[{q}]" for q in range(generator.stencil.stencil.q())])

            generator.nde(f"const auto rho = {f_eq_sum};")

    def generate_rho_inv(self, generator: 'GeneratorKernel'):
        if not generator.registered('rho_inv'):
            generator.register('rho_inv')

            # dependencies
            self.generate_rho(generator)

            # generate
            generator.nde('const auto rho_inv = 1.0 / rho;')

    def generate_u(self, generator: 'GeneratorKernel'):
        if not generator.registered('u'):
            generator.register('u')

            # dependencies
            generator.stencil.generate_d(generator)
            generator.stencil.generate_e(generator)

            if generator.stencil.stencil.d() > 1:
                self.generate_rho_inv(generator)

            # generate
            div_rho = ' * rho_inv' if generator.stencil.stencil.d() > 1 else ' / rho'

            generator.nde(f"const scalar_t u[d]{{")
            for d in range(generator.stencil.stencil.d()):
                summands = []
                for q in range(generator.stencil.stencil.q()):
                    summands.append(f"e[{q}][{d}] * f_reg[{q}]")
                generator.nde(f"    ({' + '.join(summands)})" + div_rho + ',')
            generator.nde('};')
            generator.nde()
