/* C204 adds an explicit visibility barrier for no-map reserved DDR mappings.
 * These mappings are CPU-uncached; this is not a substitute for flushing a
 * cached alias. The allocator must validate the kernel no-map reservation.
 */
#include "executor.c"

void cosmos_memory_barrier(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#elif defined(__arm__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}
