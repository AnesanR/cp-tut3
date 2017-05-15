import numpy as np
import matplotlib.pyplot as plt


class NbodyStarter:
    def __init__(self, N=500, soften=0.1, m=1.0, dt=0.01):
        self.options = {}
        self.options['G'] = 1.0
        self.options['soften'] = soften
        self.options['N'] = N
        self.options['dt'] = dt
        self.x = np.random.randn(self.options['N'])
        self.y = np.random.randn(self.options['N'])
        self.m = np.ones(self.options['N']) * (m / self.options['N'])
        self.vx = np.zeros(self.options['N'])
        self.vy = np.zeros(self.options['N'])

    def force(self):

        self.fx = np.zeros(self.options['N'])
        self.fy = np.zeros(self.options['N'])
        potential = np.zeros(self.options['N'])

        for i in range(0, self.options['N'] - 1):
            for j in range(i + 1, self.options['N']):
                xarr = self.x[i] - self.x[j]
                yarr = self.y[i] - self.y[j]
                radiusSquared = xarr ** 2 + yarr ** 2
                softened = self.options['soften'] ** 2

                if radiusSquared < softened:
                    radiusSquared = softened

                radius = np.sqrt(radiusSquared)
                newRad = radius * radiusSquared

                self.fx[i] = self.fx[i] - xarr * (1.0 / newRad) * self.m[j]
                self.fy[i] = self.fy[i] - yarr * (1.0 / newRad) * self.m[j]
                self.fx[j] = self.fx[j] + xarr * (1.0 / newRad) * self.m[i]
                self.fy[j] = self.fy[j] + yarr * (1.0 / newRad) * self.m[i]

                potential[i] = potential[i] + self.m[i] * self.m[j] * self.options['G'] * 1.0 / radius
        return potential.sum()

        def __GravPot__(self):

            potential = np.zeros(self.options['N'])
            for i in range(0, self.options['N']):
                for j in range(0, self.options['N']):
                    if (i != j):
                        xarr = self.x[i] - self.x[j]
                        yarr = self.y[i] - self.y[j]
                        radius = np.sqrt(
                            xarr ** 2 + yarr ** 2)
                        potential[i] = potential[i] + self.m[i] * self.m[j] * self.options['G'] * 1.0 / radius
            return potential

    def UpdatePos(self, dt=0.01):
        self.x = self.x + self.vx * self.options['dt']
        self.y = self.y + self.vy * self.options['dt']
        potential = self.force()
        self.vx = self.vx + self.fx * self.options['dt']
        self.vy = self.vy + self.fy * self.options['dt']
        return potential.sum()


if __name__ == '__main__':

    nclass = NbodyStarter()
    print 'energy is ', nclass.force()
    plt.ion()

    potentialvar = np.zeros(100)
    kineticvar = np.zeros(100)

    dx = 0
    while (dx < 100):
        finalPotential = nclass.UpdatePos(0.05)
        plt.clf()
        kineticenergy = np.sum(nclass.m * (nclass.vx ** 2 + nclass.vy ** 2))
        print 'Total energy: ', [finalPotential, kineticenergy, kineticenergy - 2.0 * finalPotential]
        plt.plot(nclass.x, nclass.y, 'b*')
        plt.draw()
        plt.show()
        potentialvar[dx] = finalPotential
        kineticvar[dx] = kineticenergy
        dx += 1
