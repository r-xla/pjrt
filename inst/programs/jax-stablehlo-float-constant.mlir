func.func @main () -> tensor<f32> {
%1 ="stablehlo.constant"(){
value = dense<3.14000000e+00> : tensor<f32> }
:() -> (tensor<f32>)
"func.return"(%1):(tensor<f32>) -> ()
}
