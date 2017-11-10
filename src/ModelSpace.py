"""ModelSpace.py
"""
import qutip as qt


class ModelSpace:
    VACUUM = None

    def dims(self):
        """Returns the qutip dimensions of the space
        """
        return [self.nfock(m) for m in self.modes()]

    def nfock(self, mode):
        """Returns the number of Fock states associated with the given mode
        """
        raise NotImplementedError

    # -- States --
    def modes(self):
        """Returns a generator of the full set of normal single-particle
        Fock modes
        """
        raise NotImplementedError

    def states(self):
        """Returns a generator of single-particle basis states, namely all
        of the states formed by exciting one mode to the first level
        above the vacuum
        """
        raise NotImplementedError

    def kets(self):
        """Returns a generator of the kets associated with ~states~
        """
        return (self.get_ket(s) for s in self.states())

    def get_ket(self, mode):
        """Returns the ket associated with the first exited state in the
        given single-particle mode
        """
        raise NotImplementedError

    def get_nums(self, mode):
        """Returns a tuple of identifying quantum numbers for the given mode
        """
        raise NotImplementedError

    def vacuum_state(self):
        return ModelSpace.VACUUM

    def vacuum_ket(self):
        kets = []
        for m in self.modes():
            kets.append(qt.fock_dm(self.nfock(m)))
        return qt.tensor(kets)

    # -- Operators --
    def zero(self):
        """Returns the zero operator for the space
        """
        return qt.qzero(self.dims())

    def one(self):
        """Returns identity operator for the space
        """
        return qt.qeye(self.dims())

    def destroy(self, mode):
        """Returns a destruction operator for the given mode
        """
        dims_before = []
        dims_after = []
        seen = False
        for i, mode_i in enumerate(self.modes()):
            if mode_i == mode:
                seen = True
            elif not seen:
                dims_before.append(self.nfock(mode_i))
            else:
                dims_after.append(self.nfock(mode_i))
        ops = []
        if len(dims_before) > 0:
            ops.append(qt.qeye(dims_before))
        if seen:
            ops.append(qt.destroy(self.nfock(mode)))
        if len(dims_after) > 0:
            ops.append(qt.qeye(dims_after))
        return qt.tensor(ops)

    def create(self, mode):
        """Returns a creation operator for the given mode
        """
        return self.destroy(mode).dag()
