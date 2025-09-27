func.func @main () -> tensor<0xi64> {
  %0 = "stablehlo.constant" () {
  value = dense<[]> : tensor<0xi64>
  }: () -> (tensor<0xi64>)
"func.return"(%0): (tensor<0xi64>) -> ()
}
