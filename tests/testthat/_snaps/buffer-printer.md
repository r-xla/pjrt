# printer for integers

    Code
      pjrt_buffer(1L, "i8")
    Output
      PJRTBuffer<i8: 1> 
      1

---

    Code
      pjrt_buffer(1L, "i16")
    Output
      PJRTBuffer<i16: 1> 
      1

---

    Code
      pjrt_buffer(1L, "i32")
    Output
      PJRTBuffer<i32: 1> 
      1

---

    Code
      pjrt_buffer(1L, "ui8")
    Output
      PJRTBuffer<ui8: 1> 
      1

---

    Code
      pjrt_buffer(1L, "ui16")
    Output
      PJRTBuffer<ui16: 1> 
      1

---

    Code
      pjrt_buffer(1L, "ui32")
    Output
      PJRTBuffer<ui32: 1> 
      1

---

    Code
      pjrt_buffer(c(1L, 2L, 3L))
    Output
      PJRTBuffer<i32: 3> 
      1
      2
      3

---

    Code
      pjrt_buffer(matrix(1:10, nrow = 2))
    Output
      PJRTBuffer<i32: 2,5> 
       1  3  5  7  9
       2  4  6  8 10

# printer for integers with large difference

    Code
      pjrt_buffer(x)
    Output
      PJRTBuffer<i32: 2,2> 
      1.0000e+07 3.0000e+07
      2.0000e+07          1

# inf, nan

    Code
      pjrt_buffer(c(Inf, -Inf, NaN, 1))
    Output
      PJRTBuffer<f32: 4> 
         inf
        -inf
         nan
      1.0000

---

    Code
      pjrt_buffer(c(Inf, Inf, NA, 100011.234567))
    Output
      PJRTBuffer<f32: 4> 
              inf
              inf
              nan
      100011.2344

---

    Code
      pjrt_buffer(c(Inf, Inf, NA, 100011.234567), shape = c(1, 4))
    Output
      PJRTBuffer<f32: 1,4> 
              inf         inf         nan 100011.2344

# Up to 6 digits are printed for integers

    Code
      pjrt_buffer(1234567L)
    Output
      PJRTBuffer<i32: 1> 
      1.2346e+06

---

    Code
      pjrt_buffer(-1234567L)
    Output
      PJRTBuffer<i32: 1> 
      -1.2346e+06

---

    Code
      pjrt_buffer(123456L)
    Output
      PJRTBuffer<i32: 1> 
      123456

---

    Code
      pjrt_buffer(-123456L)
    Output
      PJRTBuffer<i32: 1> 
      -123456

# printer for doubles

    Code
      pjrt_buffer(1:10, "f32")
    Output
      PJRTBuffer<f32: 10> 
       1.0000
       2.0000
       3.0000
       4.0000
       5.0000
       6.0000
       7.0000
       8.0000
       9.0000
      10.0000

---

    Code
      pjrt_buffer(1:10, "f64")
    Output
      PJRTBuffer<f64: 10> 
       1.0000
       2.0000
       3.0000
       4.0000
       5.0000
       6.0000
       7.0000
       8.0000
       9.0000
      10.0000

---

    Code
      pjrt_buffer(1e-10)
    Output
      PJRTBuffer<f32: 1> 
      1e-10 *
      1.0000

---

    Code
      pjrt_buffer(c(1e+10, 1e-10))
    Output
      PJRTBuffer<f32: 2> 
      1.0000e+10
      1.0000e-10

# printer for arrays with many dimensions

    Code
      pjrt_buffer(1:20, shape = c(1, 1, 1, 1, 1, 5, 4))
    Output
      PJRTBuffer<i32: 1,1,1,1,1,5,4> 
      (1,1,1,1,1,.,.) =
       1  6 11 16
       2  7 12 17
       3  8 13 18
       4  9 14 19
       5 10 15 20

# printer for arrays with many elements

    Code
      pjrt_buffer(1:20000, shape = c(100, 200))
    Output
      PJRTBuffer<i32: 100,200> 
      Columns 1 to 17
         1  101  201  301  401  501  601  701  801  901 1001 1101 1201 1301 1401 1501 1601
         2  102  202  302  402  502  602  702  802  902 1002 1102 1202 1302 1402 1502 1602
         3  103  203  303  403  503  603  703  803  903 1003 1103 1203 1303 1403 1503 1603
         4  104  204  304  404  504  604  704  804  904 1004 1104 1204 1304 1404 1504 1604
         5  105  205  305  405  505  605  705  805  905 1005 1105 1205 1305 1405 1505 1605
         6  106  206  306  406  506  606  706  806  906 1006 1106 1206 1306 1406 1506 1606
         7  107  207  307  407  507  607  707  807  907 1007 1107 1207 1307 1407 1507 1607
         8  108  208  308  408  508  608  708  808  908 1008 1108 1208 1308 1408 1508 1608
         9  109  209  309  409  509  609  709  809  909 1009 1109 1209 1309 1409 1509 1609
        10  110  210  310  410  510  610  710  810  910 1010 1110 1210 1310 1410 1510 1610
        11  111  211  311  411  511  611  711  811  911 1011 1111 1211 1311 1411 1511 1611
        12  112  212  312  412  512  612  712  812  912 1012 1112 1212 1312 1412 1512 1612
        13  113  213  313  413  513  613  713  813  913 1013 1113 1213 1313 1413 1513 1613
        14  114  214  314  414  514  614  714  814  914 1014 1114 1214 1314 1414 1514 1614
        15  115  215  315  415  515  615  715  815  915 1015 1115 1215 1315 1415 1515 1615
        16  116  216  316  416  516  616  716  816  916 1016 1116 1216 1316 1416 1516 1616
        17  117  217  317  417  517  617  717  817  917 1017 1117 1217 1317 1417 1517 1617
        18  118  218  318  418  518  618  718  818  918 1018 1118 1218 1318 1418 1518 1618
        19  119  219  319  419  519  619  719  819  919 1019 1119 1219 1319 1419 1519 1619
        20  120  220  320  420  520  620  720  820  920 1020 1120 1220 1320 1420 1520 1620
        21  121  221  321  421  521  621  721  821  921 1021 1121 1221 1321 1421 1521 1621
        22  122  222  322  422  522  622  722  822  922 1022 1122 1222 1322 1422 1522 1622
        23  123  223  323  423  523  623  723  823  923 1023 1123 1223 1323 1423 1523 1623
        24  124  224  324  424  524  624  724  824  924 1024 1124 1224 1324 1424 1524 1624
        25  125  225  325  425  525  625  725  825  925 1025 1125 1225 1325 1425 1525 1625
        26  126  226  326  426  526  626  726  826  926 1026 1126 1226 1326 1426 1526 1626
        27  127  227  327  427  527  627  727  827  927 1027 1127 1227 1327 1427 1527 1627
        28  128  228  328  428  528  628  728  828  928 1028 1128 1228 1328 1428 1528 1628
        29  129  229  329  429  529  629  729  829  929 1029 1129 1229 1329 1429 1529 1629
      ... [output was truncated, set n = -1 to see all]

# column width is determined per slice

    Code
      pjrt_buffer(x, shape = c(2, 2, 2))
    Output
      PJRTBuffer<f32: 2,2,2> 
      (1,.,.) =
      1.0000 3.0000
      2.0000 4.0000
      
      (2,.,.) =
      100.0000 300.0000
      200.0000 400.0000

# 1d vector

    Code
      pjrt_buffer(1:50L)
    Output
      PJRTBuffer<i32: 50> 
       1
       2
       3
       4
       5
       6
       7
       8
       9
      10
      11
      12
      13
      14
      15
      16
      17
      18
      19
      20
      21
      22
      23
      24
      25
      26
      27
      28
      29
      30
      ... [output was truncated, set n = -1 to see all]

---

    Code
      pjrt_buffer(as.double(1:50))
    Output
      PJRTBuffer<f32: 50> 
       1.0000
       2.0000
       3.0000
       4.0000
       5.0000
       6.0000
       7.0000
       8.0000
       9.0000
      10.0000
      11.0000
      12.0000
      13.0000
      14.0000
      15.0000
      16.0000
      17.0000
      18.0000
      19.0000
      20.0000
      21.0000
      22.0000
      23.0000
      24.0000
      25.0000
      26.0000
      27.0000
      28.0000
      29.0000
      30.0000
      ... [output was truncated, set n = -1 to see all]

# pjrt_buffer print integers and logicals correctly

    Code
      print(buf_int)
    Output
      PJRTBuffer<i32: 2,2> 
      -12  45
        3  -7

---

    Code
      print(buf_log)
    Output
      PJRTBuffer<pred: 2,2> 
       true  true
      false false

# printer shows last two dims as matrix for high-rank arrays

    Code
      print(buf)
    Output
      PJRTBuffer<i32: 1,1,1,1,1,5,4> 
      (1,1,1,1,1,.,.) =
       1  6 11 16
       2  7 12 17
       3  8 13 18
       4  9 14 19
       5 10 15 20

# alignment is as expected

    Code
      pjrt_buffer(c(1000L, 1L, 10L, 100L), shape = c(1, 4))
    Output
      PJRTBuffer<i32: 1,4> 
      1000    1   10  100

# wide arrays

    Code
      pjrt_buffer(1:100, shape = c(1, 2, 50))
    Output
      PJRTBuffer<i32: 1,2,50> 
      (1,.,.) =
      Columns 1 to 28
       1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55
       2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56
      Columns 29 to 49
      57 59 61 63 65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95 97
      58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
      Columns 50 to 50
       99
      100
      ... [output was truncated, set n = -1 to see all]

