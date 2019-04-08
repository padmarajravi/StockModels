from ortools.sat.python import cp_model
import numpy as np


class HotelPartialSolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, shifts, num_head_chefs, num_sous_chefs,num_foh, num_waiter, num_days, num_shifts, sols):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._shifts = shifts
        self._num_days= num_days
        self._num_head_chefs = num_head_chefs
        self._num_sous_chefs = num_sous_chefs
        self._num_foh = num_foh
        self._num_waiter = num_waiter
        self._num_shifts = num_shifts
        self._solutions = set(sols)
        self._solution_count = 0

    def on_solution_callback(self):
        self._solution_count += 1
        if self._solution_count in self._solutions:
            print('Solution %i' % self._solution_count)
            for d in range(self._num_days):
                print('Day %i' % d)
                for s in range(self._num_shifts):
                   print("shift:"+str(s))
                   for hc in range(self._num_head_chefs):
                        for sc in range(self._num_sous_chefs):
                            for f in range(self._num_foh):
                                for w in range(self._num_waiter):
                                    if self.Value(self._shifts[(d, s,hc,sc,f,w)]):
                                        print("Found")

            print()

    def solution_count(self):
        return self._solution_count


if __name__ == '__main__':
    num_shifts     = 6
    num_days       = 3
    num_head_chefs = 6
    num_sous_chefs = 6
    num_foh        = 6
    num_waiter     = 6
    matrix_shape = (num_head_chefs,num_sous_chefs,num_foh,num_waiter,num_days,num_shifts)
    model = cp_model.CpModel()
    myvars = [model.NewIntVar(0,1, 'x'+str(i)) for i in range(num_shifts*num_days*num_head_chefs*num_sous_chefs*num_foh*num_waiter)]
    shifts = np.array(myvars)
    shifts.shape = matrix_shape
    print((shifts[1,:,:,:,1,:]))
    #One person should work in one, and only one shift a day
    for i in range(6):
        for j in range(3):
            model.Add(np.sum(shifts[i,:,:,:,j,:])  == 1 )
            model.Add(np.sum(shifts[:,i,:,:,j,:])  == 1 )
            model.Add(np.sum(shifts[:,:,i,:,j,:])  == 1 )
            model.Add(np.sum(shifts[:,:,:,i,j,:])  == 1 )

    # Same pair exclusion

    """

    for i in range(6):
        for j in range(6):
            model.Add(np.sum(shifts[i,j,:,:,:,:])  == 1 )
            model.Add(np.sum(shifts[i,:,j,:,:,:])  == 1 )
            model.Add(np.sum(shifts[i,:,:,j,:,:])  == 1 )
            model.Add(np.sum(shifts[:,i,j,:,:,:])  == 1 )
            model.Add(np.sum(shifts[:,i,:,j,:,:])  == 1 )
            model.Add(np.sum(shifts[:,:,i,j,:,:])  == 1 )

   """

    solver = cp_model.CpSolver()
    solution_printer = HotelPartialSolutionPrinter(
    shifts, num_head_chefs, num_sous_chefs,num_foh, num_waiter, num_days, num_shifts, range(2))
    solver.SearchForAllSolutions(model, solution_printer)




    print("Done")














