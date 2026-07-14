#include "postgres.h"

#include "fmgr.h"

PG_MODULE_MAGIC;

void HydexInit(void);
void VectorLinearInit(void);

PGDLLEXPORT void _PG_init(void);

void
_PG_init(void)
{
	HydexInit();
	VectorLinearInit();
}
