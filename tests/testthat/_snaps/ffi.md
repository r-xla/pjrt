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
      CustomTail

# print handler supports empty header

    Code
      invisible(pjrt_execute(program, buf))
    Output
       1
       2
       3
      [ F32{3} ]

# print handler supports no head and no tail

    Code
      invisible(pjrt_execute(program, buf))
    Output
       1
       2
       3

