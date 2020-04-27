static char help[] = "Runs 4D-VAR to test my ideas about background covariance matrix construction.\n";
#include <var.h>


struct _lorenz_63_User {

  PetscReal a, b, r;
  PetscReal next_output, tf;
  
  Mat       Jac;
  Vec       X, lambda[3];
};

typedef struct _lorenz_63_User *ODEInfo;


static PetscErrorCode Lorenz63RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  PetscErrorCode    ierr;
  ODEInfo           user = (ODEInfo)ctx;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);

  f[0] = user->a * (x[1] - x[0]);
  f[1] = user->r * x[0] - x[1] - x[0] * x[2];
  f[2] = x[0] * x[1] - user->b * x[2];

  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
				      
  PetscFunctionReturn(0);
}

static PetscErrorCode Lorenz63RHSJacobian(TS ts, PetscReal t, Vec X, Mat Jac, Mat Pre, void *ctx)
{
  PetscErrorCode    ierr;
  ODEInfo           user = (ODEInfo)ctx;
  PetscInt          rows_and_cols[] = {0,1,2};
  PetscScalar       J[3][3];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  J[0][0] = -1.0 * user->a;
  J[0][1] = user->a;
  J[0][2] = 0.0;

  J[1][0] = user->r - x[2];
  J[1][1] = -1.0;
  J[1][2] = -x[0];

  J[2][0] = x[1];
  J[2][1] = x[0];
  J[2][2] = -1.0 * user->b;

  ierr = MatSetValues(Jac, 3, rows_and_cols, 3, rows_and_cols, &J[0][0], INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if(Pre != Jac){
    ierr = MatAssemblyBegin(Pre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Pre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int main(int argc, char **argv)
{
  VarInfo                  VARRunner;
  TS                       ts;
  struct _lorenz_63_User   ode_ctx;
  Vec                      obs, TrueX0;
  PetscInt                 i, j, N, nobs, nobs_ts, nx, ny, nz, hrsum,
                           *h_num_rows, *hs_nz_cols,
                           *obs_x, *obs_y, *obs_z;
  PetscReal                dt, var_window, *obs_times, a, b, r,x0t, y0t, z0t;
  PetscScalar              *x;
  PetscBool                flg;
  PetscErrorCode           ierr;
  
  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Initialized");
  /* system dimension, number of obs, var window length */
  N = 3;
  ierr = PetscOptionsGetInt(NULL, NULL, "-obs_per_window", &nobs, &flg);CHKERRQ(ierr);
  if(!flg){
    nobs = 1;
  }

  ierr = PetscCalloc5(nobs, &obs_times,
		      nobs, &obs_x,
		      nobs, &obs_y,
		      nobs, &obs_z,
		      nobs, &h_num_rows);CHKERRQ(ierr);
  
  ierr = PetscOptionsGetReal(NULL, NULL, "-window_len", &var_window, &flg);CHKERRQ(ierr);
  if(!flg){
    ierr = PetscOptionsGetReal(NULL, NULL, "-var_window", &var_window, &flg);CHKERRQ(ierr);
    if(!flg){
      var_window = 10.0;
    }
  }

  nobs_ts = nobs;
  ierr = PetscOptionsGetRealArray(NULL, NULL, "-obs_times", obs_times, &nobs_ts, &flg);CHKERRQ(ierr);
  if(flg){
    if(nobs_ts != nobs){
      SETERRQ(PETSC_COMM_WORLD, 1, "Error: if you supply observation times, you must supply as many times as there are observations.");
    }
  } else {
    for(i = 0; i < nobs; ++i){
      obs_times[i] = (i+1) * var_window / ((PetscReal)nobs);
    }
  }

  nx = nobs;
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-obs_x", obs_x, &nx, &flg);CHKERRQ(ierr);
  if(!flg){
    nx = nobs;
    for(i = 0; i < nx; ++i){
      obs_x[i] = 1;
    }
  } else {
    if(nx != nobs){
      SETERRQ(PETSC_COMM_WORLD, 1, "Error: if you supply when x is observed, you must supply either a 0 (unobserved) or a 1 (observed) for every observation time.");
    }
  }

  ny = nobs;
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-obs_y", obs_y, &ny, &flg);CHKERRQ(ierr);
  if(!flg){
    ny = nobs;
    for(i = 0; i < ny; ++i){
      obs_y[i] = 1;
    }
  } else {
    if(ny != nobs){
      SETERRQ(PETSC_COMM_WORLD, 1, "Error: if you supply when y is observed, you must supply either a 0 (unobserved) or a 1 (observed) for every observation time.");
    }
  }

  nz = nobs;
  ierr = PetscOptionsGetIntArray(NULL, NULL, "-obs_z", obs_z, &nz, &flg);CHKERRQ(ierr);
  if(!flg){
    nz = nobs;
    for(i = 0; i < nz; ++i){
      obs_z[i] = 1;
    }
  } else {
    if(nz != nobs){
      SETERRQ(PETSC_COMM_WORLD, 1, "Error: if you supply when z is observed, you must supply either a 0 (unobserved) or a 1 (observed) for every observation time.");
    }
  }

  hrsum = 0;
  for(i = 0; i < nobs; ++i){
    h_num_rows[i] = obs_x[i] + obs_y[i] + obs_z[i];
    hrsum += h_num_rows[i];
  }
  ierr = PetscCalloc1(hrsum, &hs_nz_cols);CHKERRQ(ierr);
  j = 0;
  for(i = 0; i < nobs; ++i){
    if(obs_x[i]){
      hs_nz_cols[j] = 0;
      ++j;
    }
    if(obs_y[i]){
      hs_nz_cols[j] = 1;
      ++j;
    }
    if(obs_z[i]){
      hs_nz_cols[j] = 2;
      ++j;
    }
  }
  if(j != hrsum){
    PetscPrintf(PETSC_COMM_WORLD, "WARNING, j = %d, hrsum = %d.\n", j, hrsum);
  }


  ierr = PetscOptionsGetReal(NULL, NULL, "-true_x0", &x0t, &flg);CHKERRQ(ierr);
  if(!flg){
    x0t = 10.0;
  }
  ierr = PetscOptionsGetReal(NULL, NULL, "-true_y0", &y0t, &flg);CHKERRQ(ierr);
  if(!flg){
    y0t = -5.0;
  }

  ierr = PetscOptionsGetReal(NULL, NULL, "-true_z0", &z0t, &flg);CHKERRQ(ierr);
  if(!flg){
    z0t = 2.0;
  }
  
  ierr = PetscOptionsGetReal(NULL, NULL, "-a", &a, &flg);CHKERRQ(ierr);
  if(!flg){
    a = 16.0;
  }
  ierr = PetscOptionsGetReal(NULL, NULL, "-b", &b, &flg);CHKERRQ(ierr);
  if(!flg){
    b = 4.0;
  }
  ierr = PetscOptionsGetReal(NULL, NULL, "-r", &a, &flg);CHKERRQ(ierr);
  if(!flg){
    r = 45.0;
  }
  PetscPrintf(PETSC_COMM_WORLD, "got options");

  ode_ctx.a = a;
  ode_ctx.b = b;
  ode_ctx.r = r;
  ode_ctx.tf = var_window;

  /* generate "ground truth" solution */
  ierr = VecCreate(PETSC_COMM_WORLD, &TrueX0);CHKERRQ(ierr);
  ierr = VecSetSizes(TrueX0, PETSC_DECIDE, 3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(TrueX0);CHKERRQ(ierr);
  ierr = VecGetArray(TrueX0, &x);CHKERRQ(ierr);
  x[0] = x0t;
  x[1] = y0t;
  x[2] = z0t;
  ierr = VecRestoreArray(TrueX0, &x);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts, NULL, Lorenz63RHSFunction, &ode_ctx);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(ts, TSRK5F);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, var_window);CHKERRQ(ierr);
  /*ierr = TSSetFromOptions(ts);CHKERRQ(ierr);*/
  ierr = VecDuplicate(TrueX0, &obs);CHKERRQ(ierr);
  ierr = VecCopy(TrueX0, obs);CHKERRQ(ierr);
  ierr = TSSolve(ts, obs);CHKERRQ(ierr);

  /* create 4DVAR context */
  ierr = VarInfoCreate(&VARRunner, N, 0.001, var_window,
		       nobs, obs_times, h_num_rows, NULL);CHKERRQ(ierr);

  ierr = VarInfoSetRHS(VARRunner, Lorenz63RHSFunction, &ode_ctx);CHKERRQ(ierr);
  ierr = VarInfoSetRHSJacobian(VARRunner, Lorenz63RHSJacobian, &ode_ctx);CHKERRQ(ierr);
  for(i=0; i<nobs; ++i){
    ierr = VarInfoSetIIDObsCov(VARRunner, i, 0.0001);CHKERRQ(ierr);
  }

  ierr = VarInfoSetObs(VARRunner, &obs);CHKERRQ(ierr);
  
  ierr = VarInfoSetFromOptions(VARRunner);CHKERRQ(ierr);
  ierr = VarInfoSetUp(VARRunner);CHKERRQ(ierr);
  ierr = VarInfoOptimize(VARRunner);

  return 0;
}
