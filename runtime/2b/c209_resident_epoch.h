#ifndef C209_RESIDENT_EPOCH_H
#define C209_RESIDENT_EPOCH_H
#include <stdint.h>
#define C209_RESIDENT_RUNTIME_ID 0xc2099001U
#define C209_RESIDENT_MAX_GENERATION 0x7fffffffU
/* The complete wire context ALSO carries step/layer/operator/chunk. A job
 * generation owns its epoch for all19 steps; it is never a wrapped step count.
 * The high-bit namespace cannot collide with the legacy step-only epochs. */
static int c209_resident_epoch_checked(uint64_t generation,uint32_t *epoch) {
 if(!epoch||generation==0||generation>C209_RESIDENT_MAX_GENERATION)return -5;
 *epoch=0x80000000U|(uint32_t)generation;return 0;
}
#endif
