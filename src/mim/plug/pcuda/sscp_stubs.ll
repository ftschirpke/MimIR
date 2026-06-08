; SSCP stub dispatchers for MimIR pCUDA backend.
;
; MimIR's pCUDA emitter uses a parameterized form for work-item / group ID
; queries (`__acpp_sscp_get_group_id(i32 dim)`), while AdaptiveCpp's canonical
; SSCP ABI is per-dimension (`__acpp_sscp_get_group_id_x()` etc., returning
; i64). This file bridges the two: link it against MimIR-emitted device IR
; before invoking `acpp`.
;
; See: AdaptiveCpp/include/hipSYCL/sycl/libkernel/sscp/builtins/core.hpp

declare i64 @__acpp_sscp_get_group_id_x()
declare i64 @__acpp_sscp_get_group_id_y()
declare i64 @__acpp_sscp_get_group_id_z()
declare i64 @__acpp_sscp_get_local_id_x()
declare i64 @__acpp_sscp_get_local_id_y()
declare i64 @__acpp_sscp_get_local_id_z()

define i32 @__acpp_sscp_get_group_id(i32 %dim) {
entry:
  switch i32 %dim, label %d0 [i32 1, label %d1
                              i32 2, label %d2]
d0:
  %a = call i64 @__acpp_sscp_get_group_id_x()
  %ai = trunc i64 %a to i32
  ret i32 %ai
d1:
  %b = call i64 @__acpp_sscp_get_group_id_y()
  %bi = trunc i64 %b to i32
  ret i32 %bi
d2:
  %c = call i64 @__acpp_sscp_get_group_id_z()
  %ci = trunc i64 %c to i32
  ret i32 %ci
}

define i32 @__acpp_sscp_get_local_id(i32 %dim) {
entry:
  switch i32 %dim, label %d0 [i32 1, label %d1
                              i32 2, label %d2]
d0:
  %a = call i64 @__acpp_sscp_get_local_id_x()
  %ai = trunc i64 %a to i32
  ret i32 %ai
d1:
  %b = call i64 @__acpp_sscp_get_local_id_y()
  %bi = trunc i64 %b to i32
  ret i32 %bi
d2:
  %c = call i64 @__acpp_sscp_get_local_id_z()
  %ci = trunc i64 %c to i32
  ret i32 %ci
}
