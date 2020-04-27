#include "var.h"

struct _4dvar_info {
  TS          ts, quadts;
  Tao         tao;
  PetscInt    N, nobs, nsteps, *obs_steps;
  ObsDimInfo  *obs_dims;
  PetscReal   window_len, dt, tfinal, *obs_times;
  PetscScalar b0err, tot_err, *obs_err;
  Mat         Jac, LossJac, B0sqrt, B0inv, *H, *Rinv;
  Vec         LossGrad, Xt, X0, X0b, Deltax, WDx, Deltay, WDy, Hx, *Y;/* WDx is weighted Deltax */

  TSRHSFunction xprime, cost_func;
  TSRHSJacobian xprime_jac, cost_grad;

  void          *xprime_ctx, *xprime_jac_ctx;

  VarFormFunction form_func;
  VarFormGradient form_grad;
  /* first param is B0sqrt/B0Inv (respectively), second is X at the time, third is TLM jacobian at the time */
  PetscErrorCode (*B0SqrtFunc)(Vec, TS, PetscReal, VarInfo);
  PetscErrorCode (*B0InvFunc)(Mat, Vec, Mat);

  TaoType     tao_t;
  PetscBool   has_x0;
 
};


static PetscInt find_obs_id(PetscReal t, VarInfo info)
{
  PetscInt i;
  for(i = 0; i < info->nobs; ++i){
      if(PetscAbs(t - info->obs_times[i]) < 0.5 * info->dt){
	/* found closest possible solution time to an obs */
	return i;
      }
  }
  return -1;
}

static PetscErrorCode ComputeBackgroundError(Vec X, TS ts, PetscReal t, VarInfo ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  /* this is where the magic happens */
  ierr = ctx->xprime_jac(ts, t, X, ctx->B0sqrt, ctx->B0sqrt, ctx);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(ctx->B0sqrt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->B0sqrt, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecWAXPY(ctx->Deltax, -1.0, ctx->X0b, X);CHKERRQ(ierr);
  ierr = MatMult(ctx->B0sqrt, ctx->Deltax, ctx->WDx);CHKERRQ(ierr);
  ierr = VecDot(ctx->WDx, ctx->WDx, &(ctx->b0err));CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}



static PetscErrorCode VARCostIntegrand(TS ts, PetscReal t, Vec X, Vec R, VarInfo ctx)
{
  PetscErrorCode    ierr;
  PetscInt          i, obs_id=-1;
  PetscScalar       *r;
  const PetscScalar *x;
  PetscFunctionBeginUser;

  ierr = VecGetArray(R, &r);CHKERRQ(ierr);
  if(t < ctx->dt){
    ierr = ComputeBackgroundError(X,ts,t,ctx);CHKERRQ(ierr);
    r[0] = ctx->b0err / ctx->dt;/* multiply by 1/dt to undo the TS's scaling */
    /* update X0b */
    ierr = VecCopy(X, ctx->X0b);CHKERRQ(ierr);
  } else {
    obs_id = find_obs_id(t, ctx);
    if(obs_id != -1){
      /* compute (Y - H * X)^T R^{-1} (Y - H * X) */
      ierr = ComputeObsError(ctx->H[obs_id], ctx->Rinv[obs_id],
			     X, ctx->Y[obs_id], ctx->Hx,
			     ctx->Deltay, ctx->WDy,
			     &(ctx->obs_err[obs_id]));CHKERRQ(ierr);
      r[0] = ctx->obs_err[obs_id] / ctx->dt;
    } else {
      r[0] = 0.0;
    }
  }

  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}


static PetscErrorCode VARCostGradient(TS ts, PetscReal t, Vec X, Mat Grad,
				   Mat Pre, VarInfo ctx)
{
  PetscErrorCode    ierr;
  PetscInt          i, obs_id=-1, rows[] = {0};
  const PetscScalar *x, *drdu;

  PetscFunctionBeginUser;
  

  if(t < ctx->dt){
    PetscInt cols[ctx->N];
    for(i = 0; i < ctx->N; ++i){
      cols[i] = i;
    }
    ierr = ctx->xprime_jac(ts, t, X, ctx->B0sqrt, ctx->B0sqrt, ctx);CHKERRQ(ierr);
    ierr = VecWAXPY(ctx->Deltax, -1.0, ctx->X0b, X);CHKERRQ(ierr);
    ierr = MatMult(ctx->B0sqrt, ctx->Deltax, ctx->WDx);CHKERRQ(ierr);
    /* multiply by 1/dt to undo the TS's scaling */
    ierr = VecScale(ctx->WDx, 2.0/ctx->dt);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ctx->WDx, &drdu);CHKERRQ(ierr);
    ierr = MatSetValues(Grad, 1, rows, ctx->N, cols, drdu, INSERT_VALUES);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(Grad, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Grad, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ctx->WDx, &drdu);CHKERRQ(ierr);
    
  } else {
    
    obs_id = find_obs_id(t, ctx);
    if(obs_id != -1){
      PetscInt cols[ctx->obs_dims[obs_id].dim];
      for(i = 0; i < ctx->obs_dims[obs_id].dim; ++i){
	cols[i] = ctx->obs_dims[obs_id].nzcols[i];
      }
      /* compute H^T R^{-1} (Y - H * X) */
      /*PetscPrintf(PETSC_COMM_WORLD, "At time t=[%.4f], doing Hx = H[%d]*x",t, i);*/
      ierr = MatMult(ctx->H[obs_id], X, ctx->Hx);CHKERRQ(ierr);
      /*PetscPrintf(PETSC_COMM_WORLD, "did Hx = H*x");*/
      ierr = VecWAXPY(ctx->Deltay, -1.0, ctx->Hx, ctx->Y[obs_id]);CHKERRQ(ierr);
      ierr = MatMult(ctx->Rinv[obs_id], ctx->Deltay, ctx->WDy);CHKERRQ(ierr);
      /*PetscPrintf(PETSC_COMM_WORLD, "did R^{-1}*(y-Hx)");*/
      ierr = MatMultTranspose(ctx->H[obs_id], ctx->WDy, ctx->Deltay);CHKERRQ(ierr);
      /*PetscPrintf(PETSC_COMM_WORLD, "did H^T R^{-1}(y-Hx)");*/
      ierr = VecScale(ctx->Deltay, 2.0 / ctx->dt);CHKERRQ(ierr);
      ierr = VecGetArrayRead(ctx->Deltay, &drdu);CHKERRQ(ierr);
      ierr = MatSetValues(Grad, 1, rows,
			  ctx->obs_dims[obs_id].dim, cols,
			  drdu, INSERT_VALUES);
      ierr = MatAssemblyBegin(Grad, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Grad, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(ctx->Deltay, &drdu);CHKERRQ(ierr);
    } 
  }

  PetscFunctionReturn(0);
}

extern PetscErrorCode VARFormFunction(Tao tao, Vec X, PetscReal *f, void *ctx)
{
  VarInfo     info=(VarInfo)ctx;
  Vec         Jhat;
  const PetscScalar *jhat;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* reinitialize TS and cost integral */
  ierr = TSSetTime(info->ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(info->ts, 0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(info->ts, info->dt);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(info->ts, &Jhat);CHKERRQ(ierr);
  ierr = VecSet(Jhat, 0.0);CHKERRQ(ierr);

  ierr = VarInfoSetInitialCondition(info, X);CHKERRQ(ierr);
  ierr = TSGetSolution(info->ts, &(info->Xt));CHKERRQ(ierr);
  ierr = VecCopy(info->X0, info->Xt);CHKERRQ(ierr);

  ierr = TSSolve(info->ts, info->Xt);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(info->ts, &Jhat);CHKERRQ(ierr);

  ierr = VecGetArrayRead(Jhat, &jhat);CHKERRQ(ierr);
  *f = jhat[0];
  ierr = VecRestoreArrayRead(Jhat, &jhat);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode VARFormGradient(Tao tao, Vec X, Vec G, void *ctx)
{
  VarInfo     info=(VarInfo)ctx;
  Vec         Jhat, *lambda;
  PetscScalar *lptr;
  PetscInt    i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* reinitialize TS and cost integral */
  ierr = TSSetTime(info->ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(info->ts, 0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(info->ts, info->dt);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(info->ts, &Jhat);CHKERRQ(ierr);
  ierr = VecSet(Jhat, 0.0);CHKERRQ(ierr);

  ierr = VarInfoSetInitialCondition(info, X);CHKERRQ(ierr);
  ierr = TSGetSolution(info->ts, &(info->Xt));CHKERRQ(ierr);
  ierr = VecCopy(info->X0, info->Xt);CHKERRQ(ierr);

  ierr = TSSetSaveTrajectory(info->ts);CHKERRQ(ierr);

  ierr = TSSolve(info->ts, info->Xt);CHKERRQ(ierr);
  ierr = TSGetSolveTime(info->ts, &info->tfinal);CHKERRQ(ierr);
  ierr = TSGetStepNumber(info->ts, &info->nsteps);CHKERRQ(ierr);

  ierr = TSGetCostGradients(info->ts, NULL, &lambda, NULL);CHKERRQ(ierr);

  ierr = VecGetArray(lambda[0], &lptr);CHKERRQ(ierr);
  for(i = 0; i < info->N; ++i){
    lptr[i] = 0.0;
  }
  ierr = VecRestoreArray(lambda[0], &lptr);CHKERRQ(ierr);

  ierr = TSAdjointSolve(info->ts);CHKERRQ(ierr);
  ierr = VecCopy(lambda[0], G);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}
  
    
  
  
/* TODO: change h_nz_rows to be h_nz_cols, which is really what it is */
extern PetscErrorCode VarInfoCreate(VarInfo *vctx, PetscInt N, PetscReal dt,
			     PetscReal window_len, PetscInt num_obs_per_window,
			     PetscReal *obs_times, PetscInt *h_num_rows,
			     PetscInt *h_nz_rows)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k, nrow, rowstart;
  VarInfo        ctx;
  PetscFunctionBeginUser;

  ierr = PetscNew(vctx);CHKERRQ(ierr);
  ctx = *vctx;
  ctx->nsteps = window_len / dt;
  ctx->N = N;
  ctx->nobs = num_obs_per_window;
  ctx->window_len = window_len;
  ctx->dt = dt;

  ierr = TSCreate(PETSC_COMM_WORLD, &(ctx->ts));CHKERRQ(ierr);
  ierr = TSSetProblemType(ctx->ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetEquationType(ctx->ts, TS_EQ_ODE_EXPLICIT);CHKERRQ(ierr);
  ierr = TSSetType(ctx->ts, TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(ctx->ts, TSRK4);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ctx->ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetSaveTrajectory(ctx->ts);CHKERRQ(ierr);
  
  ierr = PetscMalloc3(ctx->nobs, &(ctx->obs_times),
		      ctx->nobs, &(ctx->obs_steps),
		      ctx->nobs, &(ctx->obs_err));CHKERRQ(ierr);
  for(i = 0; i < ctx->nobs; ++i){
    ctx->obs_times[i] = obs_times[i];
    ctx->obs_steps[i] = obs_times[i] / dt;
  }
  
  ierr = PetscMalloc1(ctx->nobs, &(ctx->obs_dims));CHKERRQ(ierr);
  /* handle observations dimensions */
  for(i = 0; i < ctx->nobs; ++i){
    nrow = (h_num_rows) ? h_num_rows[i] : N;
    ierr = PetscMalloc1(nrow, &(ctx->obs_dims[i].nzcols));CHKERRQ(ierr);
    rowstart = 0;
    for(k = 0; k < i; ++k){
      rowstart += (h_num_rows) ? h_num_rows[k] : N;
    }
    for(j = 0; j < nrow; ++j){
      if(h_nz_rows){
	ctx->obs_dims[i].nzcols[j] = h_nz_rows[rowstart + j];
      } else {
	ctx->obs_dims[i].nzcols[j] = j;
      }
    }
  }
  
  ierr = PetscMalloc3(ctx->nobs, &(ctx->H),
		      ctx->nobs, &(ctx->Rinv),
		      ctx->nobs, &(ctx->Y));CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->Jac));CHKERRQ(ierr);
  ierr = MatSetSizes(ctx->Jac, PETSC_DECIDE, PETSC_DECIDE, N, N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx->Jac);CHKERRQ(ierr);
  ierr = MatSetUp(ctx->Jac);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->B0sqrt));CHKERRQ(ierr);
  ierr = MatSetSizes(ctx->B0sqrt, PETSC_DECIDE, PETSC_DECIDE, N, N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx->B0sqrt);CHKERRQ(ierr);
  ierr = MatSetUp(ctx->B0sqrt);CHKERRQ(ierr);
  
  ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->LossJac));CHKERRQ(ierr);
  ierr = MatSetSizes(ctx->LossJac, PETSC_DECIDE, PETSC_DECIDE, 1, N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx->LossJac);CHKERRQ(ierr);
  ierr = MatSetUp(ctx->LossJac);CHKERRQ(ierr);

  ierr = MatCreateVecs(ctx->Jac, &ctx->LossGrad, NULL);CHKERRQ(ierr);
  
  /*ierr = MatDuplicate(ctx->Jac, MAT_SHARE_NONZERO_PATTERN, &(ctx->B0sqrt));CHKERRQ(ierr);*/

  for(i=0; i < ctx->nobs; ++i){
    if(h_num_rows){
      ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->H[i]));CHKERRQ(ierr);
      ierr = MatSetSizes(ctx->H[i], PETSC_DECIDE, PETSC_DECIDE, h_num_rows[i], N);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->Rinv[i]));CHKERRQ(ierr);
      ierr = MatSetSizes(ctx->Rinv[i], PETSC_DECIDE, PETSC_DECIDE, h_num_rows[i], h_num_rows[i]);CHKERRQ(ierr);
      ierr = MatSetFromOptions(ctx->H[i]);CHKERRQ(ierr);
      ierr = MatSetFromOptions(ctx->Rinv[i]);CHKERRQ(ierr);
      ierr = MatSetUp(ctx->H[i]);CHKERRQ(ierr);
      ierr = MatSetUp(ctx->Rinv[i]);CHKERRQ(ierr);
      ierr = MatGetIdentityObserver(ctx->H[i]);CHKERRQ(ierr);
    } else {
      ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->H[i]));CHKERRQ(ierr);
      ierr = MatSetSizes(ctx->H[i], PETSC_DECIDE, PETSC_DECIDE, N, N);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->Rinv[i]));CHKERRQ(ierr);
      ierr = MatSetSizes(ctx->Rinv[i], PETSC_DECIDE, PETSC_DECIDE, N, N);CHKERRQ(ierr);
      ierr = MatSetFromOptions(ctx->H[i]);CHKERRQ(ierr);
      ierr = MatSetFromOptions(ctx->Rinv[i]);CHKERRQ(ierr);
      ierr = MatSetUp(ctx->H[i]);CHKERRQ(ierr);
      ierr = MatSetUp(ctx->Rinv[i]);CHKERRQ(ierr);
      ierr = MatGetIdentityObserver(ctx->H[i]);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(ctx->Rinv[i], &(ctx->Y[i]), NULL);CHKERRQ(ierr);
  }
  ierr = MatCreateVecs(ctx->Jac, &(ctx->Xt), NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD, &ctx->Hx);CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->Hx, PETSC_DECIDE, N);
  ierr = VecSetFromOptions(ctx->Hx);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD, &ctx->Deltay);CHKERRQ(ierr);
  /* TODO: CHANGE THIS */
  ierr = VecSetSizes(ctx->Deltay, PETSC_DECIDE, N);
  ierr = VecSetFromOptions(ctx->Deltay);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD, &ctx->WDy);CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->WDy, PETSC_DECIDE, N);
  ierr = VecSetFromOptions(ctx->WDy);CHKERRQ(ierr);

  ierr = MatCreateVecs(ctx->B0sqrt, &(ctx->X0), NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx->B0sqrt, &(ctx->Deltax), NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx->B0sqrt, &(ctx->WDx), NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx->B0sqrt, &(ctx->X0b), NULL);CHKERRQ(ierr);

  ctx->B0SqrtFunc = ComputeBackgroundError;
  ctx->B0InvFunc = NULL;
  ctx->cost_func = (TSRHSFunction)VARCostIntegrand;
  ctx->cost_grad = (TSRHSJacobian)VARCostGradient;
  ctx->xprime_ctx = NULL;
  ctx->xprime_jac_ctx = NULL;
  ctx->xprime = NULL;
  ctx->xprime_jac = NULL;
  ctx->form_func = VARFormFunction;
  ctx->form_grad = VARFormGradient;

  ctx->tao_t = TAOLMVM;
  ctx->has_x0 = PETSC_FALSE;
  
  *vctx = ctx;
  PetscFunctionReturn(0);
}

extern PetscErrorCode VarInfoDestroy(VarInfo ctx)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBeginUser;
  ierr = PetscFree3(ctx->obs_times, ctx->obs_steps, ctx->obs_err);CHKERRQ(ierr);
  
  for(i = 0; i < ctx->nobs; ++i){
    ierr = PetscFree(ctx->obs_dims[i].nzcols);CHKERRQ(ierr);
    ierr = MatDestroy(&ctx->Rinv[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&(ctx->H[i]));CHKERRQ(ierr);
    ierr = VecDestroy(&(ctx->Y[i]));CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->obs_dims);CHKERRQ(ierr);

  ierr = MatDestroy(&(ctx->Jac));CHKERRQ(ierr);
  ierr = MatDestroy(&(ctx->LossJac));CHKERRQ(ierr);
  ierr = MatDestroy(&(ctx->B0sqrt));CHKERRQ(ierr);
  ierr = VecDestroy(&(ctx->LossGrad));CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->Xt);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->X0b);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->WDx);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->Deltax);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->WDy);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->Deltay);CHKERRQ(ierr);

  ierr = TSDestroy(&ctx->ts);CHKERRQ(ierr);
  ierr = TaoDestroy(&ctx->tao);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


extern PetscErrorCode VarInfoSetRHS(VarInfo info, TSRHSFunction rhs, void *prob_ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  info->xprime = rhs;
  info->xprime_ctx = prob_ctx;
  ierr = TSSetRHSFunction(info->ts, NULL, rhs, prob_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

extern PetscErrorCode VarInfoSetObs(VarInfo info, Vec *Y)
{
  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBeginUser;
  for(i=0; i<info->nobs; ++i){
    ierr = VecDuplicate(Y[i], &info->Y[i]);CHKERRQ(ierr);
    ierr = VecCopy(Y[i], info->Y[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
      

PetscErrorCode VarInfoSetInitialCondition(VarInfo info, Vec x0)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(info->has_x0){
    ierr = VecCopy(info->X0, info->X0b);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x0, info->X0b);CHKERRQ(ierr);
    info->has_x0 = PETSC_TRUE;
  }
  ierr = VecCopy(x0, info->X0);CHKERRQ(ierr);
  
  PetscFunctionReturn(ierr);
}

extern PetscErrorCode VarInfoSetRHSJacobian(VarInfo info, TSRHSJacobian rhs_jac, void *prob_ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  info->xprime_jac = rhs_jac;
  info->xprime_jac_ctx = prob_ctx;
  ierr = TSSetRHSJacobian(info->ts, info->Jac, info->Jac, rhs_jac, prob_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

extern PetscErrorCode VarInfoSetFromOptions(VarInfo info)
{
  PetscErrorCode ierr;
  PetscReal      dt;
  PetscFunctionBeginUser;
  ierr = TSSetFromOptions(info->ts);CHKERRQ(ierr);
  ierr = TSGetTimeStep(info->ts, &dt);CHKERRQ(ierr);
  info->dt = dt;
  PetscFunctionReturn(ierr);
}


  
extern PetscErrorCode VarInfoSetUp(VarInfo info)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = TSSetMaxTime(info->ts, info->window_len);CHKERRQ(ierr);
  ierr = TSSetTimeStep(info->ts, info->dt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(info->ts);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(info->ts, NULL, info->xprime, info->xprime_ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(info->ts, info->Jac, info->Jac, info->xprime_jac, info->xprime_jac_ctx);CHKERRQ(ierr);
  
  ierr = TSSetSolution(info->ts, info->Xt);CHKERRQ(ierr);
  ierr = TSSetCostGradients(info->ts, 1, &info->LossGrad, NULL);CHKERRQ(ierr);
  /* create quadrature TS and evaluate on the forward pass */
  ierr = TSCreateQuadratureTS(info->ts, PETSC_TRUE, &info->quadts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(info->quadts, NULL, (TSRHSFunction)VARCostIntegrand, info);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(info->quadts, info->LossJac, info->LossJac, (TSRHSJacobian)VARCostGradient, info);CHKERRQ(ierr);

  ierr = TaoCreate(PETSC_COMM_WORLD, &info->tao);CHKERRQ(ierr);
  ierr = TaoSetType(info->tao, info->tao_t);CHKERRQ(ierr);
  ierr = TaoSetObjectiveRoutine(info->tao, info->form_func, info);CHKERRQ(ierr);
  ierr = TaoSetGradientRoutine(info->tao, info->form_grad, info);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(info->tao);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(info->tao, info->X0);CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode VarInfoOptimize(VarInfo info)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = TaoSolve(info->tao);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode VarInfoGetSolution(VarInfo info, Vec optimalX0)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = VecCopy(info->X0, optimalX0);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode VarInfoSetDiagonalObsCov(VarInfo info, PetscInt obs_id, PetscScalar *variances)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = MatGetDiagonalObsCov(info->Rinv[obs_id], variances);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(info->Rinv[obs_id],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(info->Rinv[obs_id],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VarInfoSetIIDObsCov(VarInfo info, PetscInt obs_id, PetscScalar variance)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = MatGetIIDObsCov(info->Rinv[obs_id], variance);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(info->Rinv[obs_id],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(info->Rinv[obs_id],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  


