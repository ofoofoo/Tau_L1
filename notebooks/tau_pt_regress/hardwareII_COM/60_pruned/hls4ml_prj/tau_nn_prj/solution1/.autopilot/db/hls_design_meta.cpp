#include "hls_design_meta.h"
const Port_Property HLS_Design_Meta::port_props[]={
	Port_Property("ap_clk", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_rst", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_start", 1, hls_in, -1, "", "", 1),
	Port_Property("ap_done", 1, hls_out, -1, "", "", 1),
	Port_Property("ap_idle", 1, hls_out, -1, "", "", 1),
	Port_Property("ap_ready", 1, hls_out, -1, "", "", 1),
	Port_Property("model_input", 1280, hls_in, 0, "ap_none", "in_data", 1),
	Port_Property("layer19_out", 16, hls_out, 1, "ap_vld", "out_data", 1),
	Port_Property("layer19_out_ap_vld", 1, hls_out, 1, "ap_vld", "out_vld", 1),
	Port_Property("layer20_out", 16, hls_out, 2, "ap_vld", "out_data", 1),
	Port_Property("layer20_out_ap_vld", 1, hls_out, 2, "ap_vld", "out_vld", 1),
};
const char* HLS_Design_Meta::dut_name = "tau_nn";
