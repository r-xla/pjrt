# device works

    Code
      as.character(device(buf))
    Output
      [1] "CpuDevice(id=0)"

# device print

    Code
      print(device(pjrt_buffer(1)))
    Output
      <CpuDevice(id=0)>

