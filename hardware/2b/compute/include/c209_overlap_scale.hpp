#pragma once
#include "c209_five_fixed_job.hpp"

namespace c209five {
// SCL1: magic,version,scale,inverse,analytic_bound,flags,reserved,reserved.
// A new image/runtime contract owns this record; Y3 aux0 was UNUSED.
static bool load_overlap_scale(bus_t *memory,uint64_t address,float &scale,float &inverse) {
#pragma HLS INLINE off
 if(address==0||(address&31ull))return false;
 bus_t record=memory[word_address(address,WordOffset(0))];
 ap_uint<32> s=record.range(95,64),i=record.range(127,96);
 ap_uint<8> se=s.range(30,23),ie=i.range(30,23);
 bool valid=record.range(31,0)==0x53434c31u&&record.range(63,32)==1u&&
  record.range(191,160)==1u&&record.range(255,192)==0&&
  !s[31]&&!i[31]&&s.range(22,0)==0&&i.range(22,0)==0&&
  se!=0&&se!=255&&ie!=0&&ie!=255&&ap_uint<9>(se)+ie==254;
 scale=::uf(s.to_uint());inverse=::uf(i.to_uint());
 return valid;
}
}
