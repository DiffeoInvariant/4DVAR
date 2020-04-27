#include "obs.h"

PetscErrorCode MatGetIdentityObserver(Mat H)
{
  PetscErrorCode ierr;
  PetscInt       N, M, rowcol;
  PetscScalar    val=1;

  PetscFunctionBeginUser;
  ierr = MatGetLocalSize(H, &M, &N);CHKERRQ(ierr);
  if(N != M){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Must supply diagonal matrix to MatGetIdentityObserver!");
  }
  for(rowcol=0; rowcol < N; ++rowcol){
    ierr = MatSetValues(H, 1, &rowcol, 1, &rowcol, &val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatGetPartialIdentityObserver(Mat H, PetscInt nnzcols, PetscInt *local_nzcols)
{
  PetscErrorCode ierr;
  PetscInt       N, M, row;
  PetscScalar    val=1;

  PetscFunctionBeginUser;
  ierr = MatGetLocalSize(H, &M, &N);CHKERRQ(ierr);
  if(N > M){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Must supply a matrix with at least as many columns as rows (i.e. a fat matrix) to MatGetPartialIdentityObserver.");
  }

  for(row=0; row < nnzcols; ++row){
    ierr = MatSetValuesLocal(H, 1, &row, 1, &local_nzcols[row], &val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}
  

PetscErrorCode MatGetDiagonalObsCov(Mat Rinv, PetscScalar *vars)
{
  PetscErrorCode ierr;
  PetscInt       N, M, rowcol;
  
  PetscFunctionBeginUser;
  ierr = MatGetLocalSize(Rinv, &M, &N);CHKERRQ(ierr);
  if(N != M){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Must supply diagonal matrix to MatGetDiagonalObsCov!");
  }

  for(rowcol=0; rowcol < N; ++rowcol){
    vars[rowcol] = 1.0/vars[rowcol];
    ierr = MatSetValuesLocal(Rinv, 1, &rowcol, 1, &rowcol, &vars[rowcol], INSERT_VALUES);CHKERRQ(ierr);
    vars[rowcol] = 1.0/vars[rowcol];/* undo the transformation to make sure the external environment isn't affected */
  }

  ierr = MatAssemblyBegin(Rinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Rinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


PetscErrorCode MatGetIIDObsCov(Mat Rinv, PetscScalar var)
{
  PetscErrorCode ierr;
  PetscInt       N, M, rowcol;
  
  PetscFunctionBeginUser;
  /* WARNING: ONLY WORKS WITH n=1 RIGHT NOW */
  ierr = MatGetLocalSize(Rinv, &M, &N);CHKERRQ(ierr);
  if(N != M){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Must supply diagonal matrix to MatGetDiagonalObsCov!");
  }

  var = 1.0 / var;
  for(rowcol=0; rowcol < N; ++rowcol){
    ierr = MatSetValues(Rinv, 1, &rowcol, 1, &rowcol, &var, INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(Rinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Rinv, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


PetscErrorCode ComputeObsError(Mat H, Mat Rinv, Vec X, Vec Y, Vec Hx, Vec Deltay, Vec WDy, PetscScalar *obs_err)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  ierr = MatMult(H, X, Hx);CHKERRQ(ierr);
  ierr = VecWAXPY(Deltay, -1.0, Hx, Y);CHKERRQ(ierr);
  ierr = MatMult(Rinv, Deltay, WDy);CHKERRQ(ierr);
  ierr = VecDot(Deltay, WDy, obs_err);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}
    
