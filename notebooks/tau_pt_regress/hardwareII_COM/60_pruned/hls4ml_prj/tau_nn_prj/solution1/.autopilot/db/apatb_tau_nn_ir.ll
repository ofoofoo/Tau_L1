; ModuleID = '/home/ofoo/tau_nn_conversion/hardwareII_COM/60_pruned/hls4ml_prj/tau_nn_prj/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>" = type { %"struct.ap_fixed_base<16, 7, true, AP_TRN, AP_WRAP, 0>" }
%"struct.ap_fixed_base<16, 7, true, AP_TRN, AP_WRAP, 0>" = type { %"struct.ssdm_int<16, true>" }
%"struct.ssdm_int<16, true>" = type { i16 }

; Function Attrs: inaccessiblemem_or_argmemonly noinline
define void @apatb_tau_nn_ir(%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="80" %model_input, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="1" %layer19_out, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="1" %layer20_out) local_unnamed_addr #0 {
entry:
  %model_input_copy4 = alloca i1280, align 512
  %layer19_out_copy = alloca i16, align 512
  %layer20_out_copy = alloca i16, align 512
  %0 = bitcast %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %model_input to [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]*
  call void @copy_in([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* nonnull %0, i1280* nonnull align 512 %model_input_copy4, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* nonnull %layer19_out, i16* nonnull align 512 %layer19_out_copy, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* nonnull %layer20_out, i16* nonnull align 512 %layer20_out_copy)
  call void @apatb_tau_nn_hw(i1280* %model_input_copy4, i16* %layer19_out_copy, i16* %layer20_out_copy)
  call void @copy_back([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %0, i1280* %model_input_copy4, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %layer19_out, i16* %layer19_out_copy, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %layer20_out, i16* %layer20_out_copy)
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @"onebyonecpy_hls.p0struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"(%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias "fpga.caller.interfaces"="layout_transformed", i16* noalias readonly align 512 "fpga.caller.interfaces"="layout_transformed") unnamed_addr #1 {
entry:
  %2 = icmp eq %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %0, null
  %3 = icmp eq i16* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  %.01.0.05 = getelementptr %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %0, i32 0, i32 0, i32 0, i32 0
  %5 = load i16, i16* %1, align 512
  store i16 %5, i16* %.01.0.05, align 512
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal fastcc void @"onebyonecpy_hls.p0struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>.250"(i16* noalias align 512 "fpga.caller.interfaces"="layout_transformed", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias readonly "fpga.caller.interfaces"="layout_transformed") unnamed_addr #1 {
entry:
  %2 = icmp eq i16* %0, null
  %3 = icmp eq %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  %.0.0.04 = getelementptr %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %1, i32 0, i32 0, i32 0, i32 0
  %5 = load i16, i16* %.0.0.04, align 2
  store i16 %5, i16* %0, align 512
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @"onebyonecpy_hls.p0a80struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"(i1280* noalias align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0", [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* noalias readonly "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1") #1 {
entry:
  %2 = icmp eq i1280* %0, null
  %3 = icmp eq [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  %.promoted = load i1280, i1280* %0, align 512
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %5 = phi i1280 [ %.promoted, %copy ], [ %14, %for.loop ]
  %for.loop.idx7 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"], [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %1, i64 0, i64 %for.loop.idx7, i32 0, i32 0, i32 0
  %6 = mul nuw nsw i64 16, %for.loop.idx7
  %7 = load i16, i16* %src.addr.0.0.05, align 2
  %8 = zext i64 %6 to i1280
  %9 = shl i1280 65535, %8
  %10 = zext i16 %7 to i1280
  %11 = shl i1280 %10, %8
  %12 = xor i1280 %9, -1
  %13 = and i1280 %5, %12
  %14 = or i1280 %13, %11
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx7, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 80
  br i1 %exitcond, label %for.loop, label %ret.loopexit

ret.loopexit:                                     ; preds = %for.loop
  store i1280 %14, i1280* %0, align 512
  br label %ret

ret:                                              ; preds = %ret.loopexit, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @copy_in([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="0", i1280* noalias align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias readonly "orig.arg.no"="2", i16* noalias align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias readonly "orig.arg.no"="4", i16* noalias align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="5") #2 {
entry:
  call void @"onebyonecpy_hls.p0a80struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"(i1280* align 512 %1, [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %0)
  call fastcc void @"onebyonecpy_hls.p0struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>.250"(i16* align 512 %3, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %2)
  call fastcc void @"onebyonecpy_hls.p0struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>.250"(i16* align 512 %5, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %4)
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @"onebyonecpy_hls.p0a80struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>.261"([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* noalias "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="0", i1280* noalias readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1") #1 {
entry:
  %2 = icmp eq [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %0, null
  %3 = icmp eq i1280* %1, null
  %4 = or i1 %2, %3
  br i1 %4, label %ret, label %copy

copy:                                             ; preds = %entry
  %5 = load i1280, i1280* %1, align 512
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %copy
  %for.loop.idx7 = phi i64 [ 0, %copy ], [ %for.loop.idx.next, %for.loop ]
  %6 = mul nuw nsw i64 16, %for.loop.idx7
  %dst.addr.0.0.06 = getelementptr [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"], [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %0, i64 0, i64 %for.loop.idx7, i32 0, i32 0, i32 0
  %7 = zext i64 %6 to i1280
  %8 = lshr i1280 %5, %7
  %9 = trunc i1280 %8 to i16
  store i16 %9, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx7, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, 80
  br i1 %exitcond, label %for.loop, label %ret

ret:                                              ; preds = %for.loop, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse
define internal void @copy_out([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0", i1280* noalias readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias "orig.arg.no"="2", i16* noalias readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias "orig.arg.no"="4", i16* noalias readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="5") #3 {
entry:
  call void @"onebyonecpy_hls.p0a80struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>.261"([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %0, i1280* align 512 %1)
  call fastcc void @"onebyonecpy_hls.p0struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"(%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %2, i16* align 512 %3)
  call fastcc void @"onebyonecpy_hls.p0struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"(%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %4, i16* align 512 %5)
  ret void
}

declare void @apatb_tau_nn_hw(i1280*, i16*, i16*)

; Function Attrs: argmemonly noinline norecurse
define internal void @copy_back([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0", i1280* noalias readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="1", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias "orig.arg.no"="2", i16* noalias readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="3", %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* noalias "orig.arg.no"="4", i16* noalias readonly align 512 "fpga.caller.interfaces"="layout_transformed" "orig.arg.no"="5") #3 {
entry:
  call fastcc void @"onebyonecpy_hls.p0struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"(%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %2, i16* align 512 %3)
  call fastcc void @"onebyonecpy_hls.p0struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"(%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %4, i16* align 512 %5)
  ret void
}

define void @tau_nn_hw_stub_wrapper(i1280*, i16*, i16*) #4 {
entry:
  %3 = alloca [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]
  %4 = alloca %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"
  %5 = alloca %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"
  call void @copy_out([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %3, i1280* %0, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %4, i16* %1, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %5, i16* %2)
  %6 = bitcast [80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %3 to %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"*
  call void @tau_nn_hw_stub(%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %6, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %4, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %5)
  call void @copy_in([80 x %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"]* %3, i1280* %0, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %4, i16* %1, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"* %5, i16* %2)
  ret void
}

declare void @tau_nn_hw_stub(%"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"*, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"*, %"struct.ap_fixed<16, 7, AP_TRN, AP_WRAP, 0>"*)

attributes #0 = { inaccessiblemem_or_argmemonly noinline "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #2 = { argmemonly noinline norecurse "fpga.wrapper.func"="copyin" }
attributes #3 = { argmemonly noinline norecurse "fpga.wrapper.func"="copyout" }
attributes #4 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
