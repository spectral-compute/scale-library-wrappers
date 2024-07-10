#include "cusparse/export.h"

// Override to make everything compile in this TU
#define GPUSOLVER_EXPORT_C extern "C" GPUSOLVER_EXPORT 

#define SOLVER_INLINE_EVERYTHING

// Override to compile uninlined functions
#define BODY(X) { X }

#include "cusolverDn.h"
#include "cusolverMg.h"
#include "cusolverRf.h"
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
