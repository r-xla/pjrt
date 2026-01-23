# print handler prints input tensors

    Code
      invisible(pjrt_execute(program, buf_f32, buf_i32, buf_pred))
    Output
      PJRTBuffer
       1
       2
       3
       4
      [ F32{4} ]
      PJRTBuffer
       5
       6
       7
       8
      [ S32{4} ]
      TestBuffer
       1
       0
       1
       0
      [ PRED{4} ]

