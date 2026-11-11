#include "c209_gelu_command_rtl.hpp"

// C-simulation arithmetic oracle. This is not the production implementation:
// c209_hls.tcl must register the JSON, and ip-prepare rejects a missing or
// substituted RTL leaf. Unsigned wrap is intentional, including columns+7.
void c209_gelu_command_rtl(hls::stream<ap_uint<65>>& requests,
                           hls::stream<ap_uint<34>>& commands) {
    static_assert(sizeof(unsigned)==4,"GELU command ABI requires unsigned32");
    ap_uint<65> request=requests.read();
    unsigned rows=request.range(31,0).to_uint();
    unsigned columns=request.range(63,32).to_uint();
    unsigned count=rows*((columns+7u)/8u)*4u;
    ap_uint<34> command=0;
    command.range(31,0)=count;
    command[32]=request[64];
    commands.write(command);
}
