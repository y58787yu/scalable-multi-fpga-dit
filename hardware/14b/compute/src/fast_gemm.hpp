#pragma once
#include <hls_streamofblocks.h>

// Include after MacCommand, cmd_stream, input_stream and result_stream are
// declared. The existing, SINGLE mac_service remains the owner of all DSPs
// and of the 48-bit accumulated output tile. No multiplier lives here.
//
// GEMM is now a set of concurrently running processes:
//   A DDR -> A ping/pong blocks -> A panel stream -------+
//   B DDR -> B ping/pong blocks -> weight stream -------+-> mac_service
//   tile/K commands ----------------------------------+      |
//   output metadata ----------------------------------------+-> epilogue -> DDR
//
// Each operand block becomes readable at write_lock scope exit, rather than
// at return of a whole-model producer. Depth TWO supplies one active block and
// one block being prepared. Block locks handle ownership without a feedback
// token cycle. Finite external FIFOs backpressure all producer processes.

struct FastGemmTileMeta {
    unsigned rb, nb, m, n, padded_n;
    bool stop;
};
using fast_gemm_meta_stream = hls::stream<FastGemmTileMeta>;
#define COSMOS_FAST_GEMM_STREAM_API 1

// The epilogue MUST consume m*padded_n/16 result words for each tile. It may
// discard padded outputs, but may not leave them in co. Scale to FP32 only
// here, after every participating K tile has accumulated in the MAC service.
static void fast_gemm_epilogue(c2::bus_t *c, c2::bus_t *x, const c2::Job &j,
 fast_gemm_meta_stream &meta, result_stream &co,
 c2fast::gelu_command_stream &gelu_commands,c2fast::nonlinear_pair_stream &gelu_input,
 c2fast::pair_output_stream &gelu_output,c2fast::gelu_ack_stream &gelu_ack);

struct FastGemmPlan {
    c2::Job job;
    unsigned end;
    c2::bus_t keep_k;
};

// C207 issues M256/N256 macro tiles. Each operand ping/pong consumes 256KiB;
// the physical engine retains 32x32 products, not a 256x256 multiplier array.
using FastGemmABlock = c2::bus_t[c2::GEMM_M * (c2::BK / 16)];
using FastGemmBBlock = c2::bus_t[c2::GEMM_N * (c2::BK / 16)];
using fast_gemm_a_blocks = hls::stream_of_blocks<FastGemmABlock, 2>;
using fast_gemm_b_blocks = hls::stream_of_blocks<FastGemmBBlock, 2>;

static unsigned fast_gemm_min(unsigned a, unsigned b) { return a < b ? a : b; }
static unsigned fast_gemm_pad_n(unsigned n) { return (n + 63u) & ~63u; }
static bool fast_gemm_kept(const FastGemmPlan &p, unsigned kb) {
#pragma HLS INLINE
    return !(p.job.flags & c2::SKIP_K) || p.keep_k[kb / c2::BK];
}

static unsigned fast_gemm_prepare(c2::bus_t *x, const c2::Job &j,
                                  FastGemmPlan &p) {
#pragma HLS INLINE off
    if (!j.m || j.m > c2::MAX_ROWS || !j.n || j.n > c2::FFN ||
        !j.k || j.k > c2::FFN || j.la < j.k || j.lb < j.k || j.lc < j.n ||
        j.la % 16 || j.lb % 16 || j.lc % c2::tensor::lanes(j.fc))
        return c2::BAD_SHAPE;
    unsigned end = j.kc ? j.k0 + j.kc : j.k;
    if (j.k0 % 16 || end > j.k || end <= j.k0) return c2::BAD_SHAPE;
    if (j.flags & c2::ADD_OUTPUT) return c2::BAD_FLAGS;
    if ((j.flags & c2::LOCAL_PARTIAL) && j.fc != 2) return c2::BAD_FLAGS;
    if ((j.flags & (c2::BIAS | c2::GELU)) &&
        ((j.flags & c2::LOCAL_PARTIAL) || j.k0 || end != j.k))
        return c2::BAD_FLAGS;
    if(j.flags & ~(c2::BIAS|c2::GELU|c2::LOCAL_PARTIAL|c2::SKIP_K|c2::WEIGHTS_INT8))
        return c2::BAD_FLAGS;
    if((j.flags&c2::WEIGHTS_INT8)&&j.lb%32)return c2::BAD_SHAPE;
    p.job = j;
    p.end = end;
    // FFN/BK <= 32, so the entire existing global K-block bitmap is one
    // 256-bit word. Read aux once BEFORE dataflow, eliminating competing aux
    // readers and independently timed mask decisions in different processes.
    p.keep_k = (j.flags & c2::SKIP_K) ? x[j.s3 >> 5] : c2::bus_t(-1);
    return c2::OK;
}

static void fast_gemm_commands(const FastGemmPlan &p, cmd_stream &commands,
                               fast_gemm_meta_stream &meta, unsigned &tiles) {
#pragma HLS INLINE off
    const c2::Job &j = p.job;
    unsigned count = 0;
    for (unsigned rb = 0; rb < j.m; rb += c2::GEMM_M)
        for (unsigned nb = 0; nb < j.n; nb += c2::GEMM_N) {
            unsigned mm = fast_gemm_min(c2::GEMM_M, j.m - rb);
            unsigned nn = fast_gemm_min(c2::GEMM_N, j.n - nb);
            unsigned np = fast_gemm_pad_n(nn);
            FastGemmTileMeta t = {rb, nb, mm, nn, np, false};
            meta.write(t);
            bool first = true;
            for (unsigned kb = j.k0; kb < p.end; kb += c2::BK) {
                if (fast_gemm_kept(p, kb)) {
                    unsigned kk = fast_gemm_min(c2::BK, p.end - kb);
                    unsigned kp = (kk + 15u) & ~15u;
                    MacCommand d = {mm, np, kp, first, false, false};
                    commands.write(d);
                    first = false;
                    count++;
                }
            }
            // Empty contribution (all K blocks skipped) still produces an
            // explicitly reset zero tile. A zero-K commit needs NO A/B block.
            MacCommand commit = {mm, np, 0, first, true, false};
            commands.write(commit);
        }
    FastGemmTileMeta eof = {0, 0, 0, 0, 0, true};
    meta.write(eof);
    tiles = count;
}

static void fast_gemm_load_a(c2::bus_t *a, const FastGemmPlan &p,
                             fast_gemm_a_blocks &blocks) {
#pragma HLS INLINE off
    const c2::Job &j = p.job;
#ifdef COSMOS_C207
    // N256 reuse plus the MAC's local A panel replaces the old 512KiB
    // K<=1024 resident copy. The two bounded stream blocks overlap DDR reads
    // with consumption without reserving a second, duplicate large cache.
    for (unsigned rb = 0; rb < j.m; rb += c2::GEMM_M) {
        unsigned mm = fast_gemm_min(c2::GEMM_M, j.m-rb);
        for (unsigned nb = 0; nb < j.n; nb += c2::GEMM_N)
            for (unsigned kb = j.k0; kb < p.end; kb += c2::BK) {
                if (fast_gemm_kept(p, kb)) {
                    unsigned kk = fast_gemm_min(c2::BK, p.end-kb);
                    unsigned words = (kk+15u)/16u;
                    hls::write_lock<FastGemmABlock> block(blocks);
                    for (unsigned r=0; r<mm; ++r)
                        for (unsigned t=0; t<words; ++t) {
#pragma HLS PIPELINE II=1
                            c2::bus_t packed = a[(j.a>>5)+((uint64_t)(rb+r)*j.la+kb)/16+t];
                            for (unsigned l=0; l<16; ++l) {
#pragma HLS UNROLL
                                if (16*t+l>=kk) packed.range(16*l+15,16*l)=0;
                            }
                            block[r*(c2::BK/16)+t]=packed;
                        }
                }
            }
    }
#else
    // A bounded 512KiB resident cache covers K<=1024 at M256. For wider
    // matrices, A is streamed again per output tile: this trades activation
    // rereads for fourfold fewer B reads at M256, with bounded BRAM use.
    const unsigned CACHE_K = 1024;
    const unsigned CACHE_WORDS = CACHE_K / 16;
    c2::bus_t resident[c2::GEMM_M * CACHE_WORDS];
#pragma HLS BIND_STORAGE variable=resident type=ram_2p impl=bram
    bool cache_enabled = j.k <= CACHE_K;
    for (unsigned rb = 0; rb < j.m; rb += c2::GEMM_M) {
        unsigned mm = fast_gemm_min(c2::GEMM_M, j.m - rb);
        if (cache_enabled) {
            for (unsigned r = 0; r < mm; r++)
                for (unsigned kb = j.k0; kb < p.end; kb += c2::BK) {
                    if (fast_gemm_kept(p, kb)) {
                        unsigned kk = fast_gemm_min(c2::BK, p.end - kb);
                        unsigned words = (kk + 15u) / 16u;
                        for (unsigned t = 0; t < words; t++) {
#pragma HLS PIPELINE II=1
                            resident[r * CACHE_WORDS + kb / 16 + t] =
                                a[(j.a >> 5) +
                                  ((uint64_t)(rb + r) * j.la + kb) / 16 + t];
                        }
                    }
                }
        }
        for (unsigned nb = 0; nb < j.n; nb += c2::GEMM_N) {
            for (unsigned kb = j.k0; kb < p.end; kb += c2::BK) {
                if (fast_gemm_kept(p, kb)) {
                    unsigned kk = fast_gemm_min(c2::BK, p.end - kb);
                    unsigned words = (kk + 15u) / 16u;
                    hls::write_lock<FastGemmABlock> block(blocks);
                    for (unsigned r = 0; r < mm; r++)
                        for (unsigned t = 0; t < words; t++) {
#pragma HLS PIPELINE II=1
                            c2::bus_t packed = cache_enabled ?
                                resident[r * CACHE_WORDS + kb / 16 + t] :
                                a[(j.a >> 5) +
                                  ((uint64_t)(rb + r) * j.la + kb) / 16 + t];
                            // A physical tail word may contain unrelated
                            // nonzero padding. Only exact logical K is used.
                            for (unsigned l = 0; l < 16; l++) {
#pragma HLS UNROLL
                                if (16 * t + l >= kk)
                                    packed.range(16 * l + 15, 16 * l) = 0;
                            }
                            block[r * (c2::BK / 16) + t] = packed;
                        }
                }
            }
        }
    }
#endif
}

static void fast_gemm_emit_a(const FastGemmPlan &p,
                             fast_gemm_a_blocks &blocks, input_stream &ain) {
#pragma HLS INLINE off
    const c2::Job &j = p.job;
    for (unsigned rb = 0; rb < j.m; rb += c2::GEMM_M)
        for (unsigned nb = 0; nb < j.n; nb += c2::GEMM_N) {
            unsigned mm = fast_gemm_min(c2::GEMM_M, j.m - rb);
            unsigned np = fast_gemm_pad_n(fast_gemm_min(c2::GEMM_N, j.n - nb));
            for (unsigned kb = j.k0; kb < p.end; kb += c2::BK) {
                if (fast_gemm_kept(p, kb)) {
                    unsigned kk = fast_gemm_min(c2::BK, p.end - kb);
                    unsigned words = (kk + 15u) / 16u;
                    hls::read_lock<FastGemmABlock> block(blocks);
#ifdef COSMOS_C207
                    c207mac::emit_packed_panel<c2::GEMM_M,c2::BK>(block,mm,words*16,ain);
#else
                    for (unsigned group = 0; group < np; group += c2::PO)
                        for (unsigned r = 0; r < mm; r++)
                            for (unsigned t = 0; t < words; t++) {
#pragma HLS PIPELINE II=1
                                ain.write(block[r * (c2::BK / 16) + t]);
                            }
#endif
                }
            }
        }
}

static void fast_gemm_load_b(c2::bus_t *b, const FastGemmPlan &p,
                             fast_gemm_b_blocks &blocks) {
#pragma HLS INLINE off
    const c2::Job &j = p.job;
    for (unsigned rb = 0; rb < j.m; rb += c2::GEMM_M)
        for (unsigned nb = 0; nb < j.n; nb += c2::GEMM_N) {
            unsigned nn = fast_gemm_min(c2::GEMM_N, j.n - nb);
            unsigned np = fast_gemm_pad_n(nn);
            for (unsigned kb = j.k0; kb < p.end; kb += c2::BK) {
                if (fast_gemm_kept(p, kb)) {
                    unsigned kk = fast_gemm_min(c2::BK, p.end - kb);
                    unsigned words = (kk + 15u) / 16u;
                    hls::write_lock<FastGemmBBlock> block(blocks);
                    for (unsigned n = 0; n < np; n++) {
                        c2::bus_t packed8=0;
                        for (unsigned t = 0; t < words; t++) {
#pragma HLS PIPELINE II=1
                            c2::bus_t packed = 0;
                            if (n < nn) {
                                if(j.flags&c2::WEIGHTS_INT8) {
                                    unsigned position=kb+16*t;
                                    if(t==0||(position&31u)==0)
                                        packed8=b[(j.b>>5)+((uint64_t)(nb+n)*j.lb+position)/32];
                                    // k_start and every word position are 16-element
                                    // aligned. Select one half once; per-lane dynamic
                                    // bit ranges would infer sixteen barrel shifters.
                                    ap_uint<128> half=(position&16u)?ap_uint<128>(packed8.range(255,128)):ap_uint<128>(packed8.range(127,0));
                                    for(unsigned l=0;l<16;l++) {
#pragma HLS UNROLL
                                        c2::i16 value=ap_int<8>(half.range(8*l+7,8*l));
                                        packed.range(16*l+15,16*l)=ap_uint<16>(value);
                                    }
                                } else {
                                    packed = b[(j.b >> 5) +
                                        ((uint64_t)(nb + n) * j.lb + kb) / 16 + t];
                                }
                            }
                            for (unsigned l = 0; l < 16; l++) {
#pragma HLS UNROLL
                                if (16 * t + l >= kk)
                                    packed.range(16 * l + 15, 16 * l) = 0;
                            }
                            // Include every padded output row expected by
                            // the unchanged service, never a stale RAM value.
                            block[n * (c2::BK / 16) + t] = packed;
                        }
                    }
                }
            }
        }
}

static void fast_gemm_emit_b(const FastGemmPlan &p,
                             fast_gemm_b_blocks &blocks, input_stream &bin) {
#pragma HLS INLINE off
    const c2::Job &j = p.job;
    for (unsigned rb = 0; rb < j.m; rb += c2::GEMM_M)
        for (unsigned nb = 0; nb < j.n; nb += c2::GEMM_N) {
            unsigned np = fast_gemm_pad_n(fast_gemm_min(c2::GEMM_N, j.n - nb));
            for (unsigned kb = j.k0; kb < p.end; kb += c2::BK) {
                if (fast_gemm_kept(p, kb)) {
                    unsigned kk = fast_gemm_min(c2::BK, p.end - kb);
                    unsigned words = (kk + 15u) / 16u;
                    hls::read_lock<FastGemmBBlock> block(blocks);
#ifdef COSMOS_C207
                    c207mac::emit_packed_panel<c2::GEMM_N,c2::BK>(block,np,words*16,bin);
#else
                    for (unsigned n = 0; n < np; n++)
                        for (unsigned t = 0; t < words; t++) {
#pragma HLS PIPELINE II=1
                            bin.write(block[n * (c2::BK / 16) + t]);
                        }
#endif
                }
            }
        }
}

// Synthesis entry point. For ordinary sequential C simulation, use the
// original GEMM math path with the same fast epilogue. To exercise this exact
// protocol natively, run producers -> mac_service -> epilogue explicitly;
// native hls streams are unbounded, but RTL channels and block pools are NOT.
static void fast_gemm_dataflow(c2::bus_t *a, c2::bus_t *b, c2::bus_t *c,
                              c2::bus_t *x, const FastGemmPlan &p,
                              cmd_stream &commands, input_stream &ain,
                              input_stream &bin, result_stream &co,
 unsigned &tiles,c2fast::gelu_command_stream &gelu_commands,
 c2fast::nonlinear_pair_stream &gelu_input,c2fast::pair_output_stream &gelu_output,
 c2fast::gelu_ack_stream &gelu_ack) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    fast_gemm_a_blocks a_blocks;
    fast_gemm_b_blocks b_blocks;
    fast_gemm_meta_stream meta;
#pragma HLS BIND_STORAGE variable=a_blocks type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=b_blocks type=ram_2p impl=bram
#pragma HLS STREAM variable=meta depth=2
    fast_gemm_commands(p, commands, meta, tiles);
    fast_gemm_load_a(a, p, a_blocks);
    fast_gemm_emit_a(p, a_blocks, ain);
    fast_gemm_load_b(b, p, b_blocks);
    fast_gemm_emit_b(p, b_blocks, bin);
    fast_gemm_epilogue(c,x,p.job,meta,co,gelu_commands,gelu_input,gelu_output,gelu_ack);
}

static unsigned fast_gemm_stage(c2::bus_t *a, c2::bus_t *b, c2::bus_t *c,
                                 c2::bus_t *x, const c2::Job &j,
                                 cmd_stream &commands, input_stream &ain,
                                 input_stream &bin, result_stream &co,
 unsigned &tiles,c2fast::gelu_command_stream &gelu_commands,
 c2fast::nonlinear_pair_stream &gelu_input,c2fast::pair_output_stream &gelu_output,
 c2fast::gelu_ack_stream &gelu_ack) {
#pragma HLS INLINE off
    FastGemmPlan p;
    unsigned status = fast_gemm_prepare(x, j, p);
    if (status != c2::OK) return status;
    fast_gemm_dataflow(a,b,c,x,p,commands,ain,bin,co,tiles,gelu_commands,gelu_input,gelu_output,gelu_ack);
    return c2::OK;
}
