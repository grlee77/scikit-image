#include <math.h>

/* A fast approximation of the exponential function.
 * Reference: https://schraudolph.org/pubs/Schraudolph99.pdf */

static union
{
    double d;
    struct
    {

#ifdef LITTLE_ENDIAN
    int j, i;
#else
    int i, j;
#endif
    } n;
} _eco;

#define EXP_A (1048576/M_LN2) /* use 1512775 for integer version */
#define EXP_C 60801           /* for min. RMS error */
/* #define EXP_C 45799 */     /* for min. max. relative error */
/* #define EXP_C 68243 */     /* for min. mean relative error */
#define EXP(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)
