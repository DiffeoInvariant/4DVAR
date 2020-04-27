#ifndef PETSC_4DVAR_OBS_H
#define PETSC_4DVAR_OBS_H
#include <petscmat.h>
#include <petscvec.h>

struct _var_obs_dim_info {
  PetscInt   dim, *nzcols;
};

typedef struct _var_obs_dim_info ObsDimInfo;


extern PetscErrorCode MatGetIdentityObserver(Mat);

/* Input params:
   pointer to int: which elements of the state (indices) are observed, and 
   in which order? for example, if the full state is 5-dimensional and you 
   want an observation operator that selects the first, third, and fifth components
   of the full state, the second parameter should be 3, and the third parameter should point to an array that looks like
   {0, 2, 4}.

   NOTE: this function calls MatSetValuesLocal, if your Mat is an MPI type, 
   be sure to pass only the LOCAL indices to keep.

   Output params:
   the matrix H (MatCreate() and MatSetSizes() MUST have already been called)
*/
extern PetscErrorCode MatGetPartialIdentityObserver(Mat, PetscInt, PetscInt *);

/* constructs a diagonal R^{-1}  with diagonal elements being the inverses of the elements
   pointed to by the second parameter. MatCreate() and MatSetSizes() MUST
   have been called first. If your matrix is parallel, only pass the local values. */
extern PetscErrorCode MatGetDiagonalObsCov(Mat, PetscScalar *);

/* constructs diagonal R^{-1} with all covariances the same. This function
   is identical to calling MatGetDiagonalObsCov with the second parameter
   pointing to an array whose values are all the same.
*/
extern PetscErrorCode MatGetIIDObsCov(Mat, PetscScalar);

/* Input parameters:
   first mat: observation operator H
   second mat: inverse error covariance R^{-1}
   first vec: state X
   second vec: observation Y
   third vec: work vector to store H*x
   fourth vec: work vector to store (Y - HX)
   fifth vec: work vector to store R^{-1}(Y - HX)
   
   Output parameters:
   pointer to PetscScalar: pointer in which to place the observation error
*/
extern PetscErrorCode ComputeObsError(Mat, Mat, Vec, Vec,
				      Vec, Vec, Vec, PetscScalar *);

#endif
