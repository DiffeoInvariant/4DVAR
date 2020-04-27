#ifndef PETSC_4DVAR_RUNNER_H
#define PETSC_4DVAR_RUNNER_H
#include <petscts.h>
#include <petsctao.h>
#include "obs.h"

typedef struct _4dvar_info *VarInfo;


typedef PetscErrorCode (*VarFormFunction)(Tao, Vec, PetscReal *, void *);
typedef PetscErrorCode (*VarFormGradient)(Tao, Vec, Vec, void *);
				     


/* N is dimension of the system, h_num_rows is the dimension of each of the obs*/
extern PetscErrorCode VarInfoCreate(VarInfo *vctx, PetscInt N, PetscReal dt,
				    PetscReal window_len, PetscInt num_obs_per_window,
				    PetscReal *obs_times, PetscInt *h_num_rows,
				    PetscInt *h_nz_rows);

extern PetscErrorCode VarInfoDestroy(VarInfo);


extern PetscErrorCode VarInfoSetRHS(VarInfo, TSRHSFunction, void *);

extern PetscErrorCode VarInfoSetRHSJacobian(VarInfo, TSRHSJacobian, void *);

extern PetscErrorCode VarInfoSetObs(VarInfo, Vec *);

extern PetscErrorCode VarInfoSetDiagonalObsCov(VarInfo, PetscInt, PetscScalar *);

extern PetscErrorCode VarInfoSetIIDObsCov(VarInfo, PetscInt, PetscScalar);

extern PetscErrorCode VarInfoSetInitialCondition(VarInfo, Vec);

extern PetscErrorCode VarInfoSetFromOptions(VarInfo);

/* call this routine AFTER setting the RHS function and jacobian and any other options, but BEFORE the optimization */
extern PetscErrorCode VarInfoSetUp(VarInfo);

extern PetscErrorCode VarInfoOptimize(VarInfo);

extern PetscErrorCode VarInfoGetSolution(VarInfo, Vec);
#endif
  
