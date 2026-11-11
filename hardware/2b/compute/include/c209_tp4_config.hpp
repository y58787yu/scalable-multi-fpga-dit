#pragma once
// Fixed Cosmos-Predict2-2B TP2 profile. SP1/SP2 share this finite compute IP.
// SP=2 describes the largest peer group; SP1 uses one4096-row partition.
// Logical profile dimensions never change the physical MAC tile or its parallelism.
namespace c209tp4 {
constexpr unsigned HARDWARE_ID=0xC2092B00u;
constexpr unsigned RUNTIME_ID=0xC2092B01u;
constexpr unsigned TOKENS=4096u;
constexpr unsigned HIDDEN=2048u;
constexpr unsigned LOCAL_WIDTH=1024u;
constexpr unsigned LOCAL_FFN=4096u;
constexpr unsigned LOCAL_HEADS=8u;
constexpr unsigned HEAD_DIM=128u;
constexpr unsigned LAYERS=28u;
constexpr unsigned LAYERS_PER_STAGE=28u;
constexpr unsigned UPDATES=19u;
constexpr unsigned TP=2u;
constexpr unsigned PP=1u;
constexpr unsigned SP=2u;
constexpr unsigned WEIGHT_GROUPS=2u;
constexpr unsigned LAYERS_PER_WEIGHT_GROUP=14u;
constexpr unsigned CROSS_KEYS=512u;
constexpr unsigned TILE_M=256u;
constexpr unsigned TILE_N=128u;
constexpr unsigned TILE_K=128u;
constexpr unsigned PHYSICAL_SHARDS=5u;
constexpr unsigned MAC_LANES_PER_SHARD=256u;
constexpr unsigned MAX_LOCAL_ROWS=4096u;
constexpr unsigned LOCAL_ROWS[SP]={2048u,2048u};
constexpr unsigned TOKEN_OFFSETS[SP]={0u,2048u};
constexpr unsigned SCALE_HEAD_STRIDE=((LOCAL_HEADS+7u)/8u)*8u;
static_assert(LOCAL_HEADS<=16 && LOCAL_WIDTH==LOCAL_HEADS*HEAD_DIM,"TP2 head geometry");
static_assert(2u*HIDDEN==LOCAL_FFN && HIDDEN%TILE_K==0,"Finite TP2 tile geometry");
} // namespace c209tp4
