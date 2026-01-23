# printer for integers

    Code
      pjrt_buffer(1L, "i8")
    Output
      PJRTBuffer 
       1
      [ CPUi8{1} ] 

---

    Code
      pjrt_buffer(1L, "i16")
    Output
      PJRTBuffer 
       1
      [ CPUi16{1} ] 

---

    Code
      pjrt_buffer(1L, "i32")
    Output
      PJRTBuffer 
       1
      [ CPUi32{1} ] 

---

    Code
      pjrt_buffer(1L, "ui8")
    Output
      PJRTBuffer 
       1
      [ CPUui8{1} ] 

---

    Code
      pjrt_buffer(1L, "ui16")
    Output
      PJRTBuffer 
       1
      [ CPUui16{1} ] 

---

    Code
      pjrt_buffer(1L, "ui32")
    Output
      PJRTBuffer 
       1
      [ CPUui32{1} ] 

---

    Code
      pjrt_buffer(c(1L, 2L, 3L))
    Output
      PJRTBuffer 
       1
       2
       3
      [ CPUi32{3} ] 

---

    Code
      pjrt_buffer(matrix(1:10, nrow = 2))
    Output
      PJRTBuffer 
        1  3  5  7  9
        2  4  6  8 10
      [ CPUi32{2x5} ] 

# printer for integers with large difference

    Code
      pjrt_buffer(x)
    Output
      PJRTBuffer 
       1.0000e+07 3.0000e+07
       2.0000e+07          1
      [ CPUi32{2x2} ] 

# inf, nan

    Code
      pjrt_buffer(c(Inf, -Inf, NaN, 1))
    Output
      PJRTBuffer 
        inf
       -inf
        nan
          1
      [ CPUf32{4} ] 

---

    Code
      pjrt_buffer(c(Inf, Inf, NA, 100011.234567))
    Output
      PJRTBuffer 
               inf
               inf
               nan
       100011.2344
      [ CPUf32{4} ] 

---

    Code
      pjrt_buffer(c(Inf, Inf, NA, 100011.234567), shape = c(1, 4))
    Output
      PJRTBuffer 
               inf         inf         nan 100011.2344
      [ CPUf32{1x4} ] 

# Up to 6 digits are printed for integers

    Code
      pjrt_buffer(1234567L)
    Output
      PJRTBuffer 
       1.2346e+06
      [ CPUi32{1} ] 

---

    Code
      pjrt_buffer(-1234567L)
    Output
      PJRTBuffer 
       -1.2346e+06
      [ CPUi32{1} ] 

---

    Code
      pjrt_buffer(123456L)
    Output
      PJRTBuffer 
       123456
      [ CPUi32{1} ] 

---

    Code
      pjrt_buffer(-123456L)
    Output
      PJRTBuffer 
       -123456
      [ CPUi32{1} ] 

# printer for doubles

    Code
      pjrt_buffer(1:10, "f32")
    Output
      PJRTBuffer 
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
      [ CPUf32{10} ] 

---

    Code
      pjrt_buffer(1:10, "f64")
    Output
      PJRTBuffer 
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
      [ CPUf64{10} ] 

---

    Code
      pjrt_buffer(c(1, 2.5), "f32")
    Output
      PJRTBuffer 
       1.0000
       2.5000
      [ CPUf32{2} ] 

---

    Code
      pjrt_buffer(1e-10)
    Output
      PJRTBuffer 
       1.0000e-10
      [ CPUf32{1} ] 

---

    Code
      pjrt_buffer(c(1e+10, 1e-10))
    Output
      PJRTBuffer 
       1.0000e+10
       1.0000e-10
      [ CPUf32{2} ] 

# integer-valued floats with truncation

    Code
      pjrt_buffer(1:50, "f32", shape = c(50, 1))
    Output
      PJRTBuffer 
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
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUf32{50x1} ] 

---

    Code
      pjrt_buffer(1:100, "f32", shape = c(2, 50))
    Output
      PJRTBuffer 
      Columns 1 to 28
        1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55
        2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56
      Columns 29 to 49
       57 59 61 63 65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95 97
       58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
      Columns 50 to 50
        99
       100
      [ CPUf32{2x50} ] 

---

    Code
      pjrt_buffer(1:24, "f32", shape = c(2, 3, 4))
    Output
      PJRTBuffer 
      (1,.,.) =
        1  7 13 19
        3  9 15 21
        5 11 17 23
      
      (2,.,.) =
        2  8 14 20
        4 10 16 22
        6 12 18 24
      [ CPUf32{2x3x4} ] 

# printer for arrays with many dimensions

    Code
      pjrt_buffer(1:20, shape = c(1, 1, 1, 1, 1, 5, 4))
    Output
      PJRTBuffer 
      (1,1,1,1,1,.,.) =
        1  6 11 16
        2  7 12 17
        3  8 13 18
        4  9 14 19
        5 10 15 20
      [ CPUi32{1x1x1x1x1x5x4} ] 

# column width is determined per slice

    Code
      pjrt_buffer(x, shape = c(2, 2, 2))
    Output
      PJRTBuffer 
      (1,.,.) =
       1 3
       2 4
      
      (2,.,.) =
       100 300
       200 400
      [ CPUf32{2x2x2} ] 

# 1d vector

    Code
      pjrt_buffer(1:50L)
    Output
      PJRTBuffer 
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
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUi32{50} ] 

---

    Code
      pjrt_buffer(as.double(1:50))
    Output
      PJRTBuffer 
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
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUf32{50} ] 

# logicals

    Code
      buf_log
    Output
      PJRTBuffer 
       1 1
       0 0
      [ CPUpred{2x2} ] 

# alignment is as expected

    Code
      pjrt_buffer(c(1000L, 1L, 10L, 100L), shape = c(1, 4))
    Output
      PJRTBuffer 
       1000    1   10  100
      [ CPUi32{1x4} ] 

# wide arrays

    Code
      pjrt_buffer(1:100, shape = c(1, 2, 50))
    Output
      PJRTBuffer 
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
      [ CPUi32{1x2x50} ] 

---

    Code
      pjrt_buffer(1:1000, shape = c(1, 2, 500))
    Output
      PJRTBuffer 
      (1,.,.) =
      Columns 1 to 28
        1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55
        2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56
      Columns 29 to 49
       57 59 61 63 65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95 97
       58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98
      Columns 50 to 70
        99 101 103 105 107 109 111 113 115 117 119 121 123 125 127 129 131 133 135 137 139
       100 102 104 106 108 110 112 114 116 118 120 122 124 126 128 130 132 134 136 138 140
      Columns 71 to 91
       141 143 145 147 149 151 153 155 157 159 161 163 165 167 169 171 173 175 177 179 181
       142 144 146 148 150 152 154 156 158 160 162 164 166 168 170 172 174 176 178 180 182
      Columns 92 to 112
       183 185 187 189 191 193 195 197 199 201 203 205 207 209 211 213 215 217 219 221 223
       184 186 188 190 192 194 196 198 200 202 204 206 208 210 212 214 216 218 220 222 224
      Columns 113 to 133
       225 227 229 231 233 235 237 239 241 243 245 247 249 251 253 255 257 259 261 263 265
       226 228 230 232 234 236 238 240 242 244 246 248 250 252 254 256 258 260 262 264 266
      Columns 134 to 154
       267 269 271 273 275 277 279 281 283 285 287 289 291 293 295 297 299 301 303 305 307
       268 270 272 274 276 278 280 282 284 286 288 290 292 294 296 298 300 302 304 306 308
      Columns 155 to 175
       309 311 313 315 317 319 321 323 325 327 329 331 333 335 337 339 341 343 345 347 349
       310 312 314 316 318 320 322 324 326 328 330 332 334 336 338 340 342 344 346 348 350
      Columns 176 to 196
       351 353 355 357 359 361 363 365 367 369 371 373 375 377 379 381 383 385 387 389 391
       352 354 356 358 360 362 364 366 368 370 372 374 376 378 380 382 384 386 388 390 392
      Columns 197 to 217
       393 395 397 399 401 403 405 407 409 411 413 415 417 419 421 423 425 427 429 431 433
       394 396 398 400 402 404 406 408 410 412 414 416 418 420 422 424 426 428 430 432 434
      Columns 218 to 238
       435 437 439 441 443 445 447 449 451 453 455 457 459 461 463 465 467 469 471 473 475
       436 438 440 442 444 446 448 450 452 454 456 458 460 462 464 466 468 470 472 474 476
      Columns 239 to 259
       477 479 481 483 485 487 489 491 493 495 497 499 501 503 505 507 509 511 513 515 517
       478 480 482 484 486 488 490 492 494 496 498 500 502 504 506 508 510 512 514 516 518
      Columns 260 to 280
       519 521 523 525 527 529 531 533 535 537 539 541 543 545 547 549 551 553 555 557 559
       520 522 524 526 528 530 532 534 536 538 540 542 544 546 548 550 552 554 556 558 560
      Columns 281 to 301
       561 563 565 567 569 571 573 575 577 579 581 583 585 587 589 591 593 595 597 599 601
       562 564 566 568 570 572 574 576 578 580 582 584 586 588 590 592 594 596 598 600 602
      Columns 302 to 322
       603 605 607 609 611 613 615 617 619 621 623 625 627 629 631 633 635 637 639 641 643
       604 606 608 610 612 614 616 618 620 622 624 626 628 630 632 634 636 638 640 642 644
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUi32{1x2x500} ] 

# scalar

    Code
      pjrt_scalar(1, "f32")
    Output
      PJRTBuffer 
       1
      [ CPUf32{} ] 

---

    Code
      pjrt_scalar(-10.1213, "f64")
    Output
      PJRTBuffer 
       -10.1213
      [ CPUf64{} ] 

---

    Code
      pjrt_scalar(10^6, "f32")
    Output
      PJRTBuffer 
       1.0000e+06
      [ CPUf32{} ] 

---

    Code
      pjrt_scalar(-10^6, "f32")
    Output
      PJRTBuffer 
       -1.0000e+06
      [ CPUf32{} ] 

---

    Code
      pjrt_scalar(10^5, "f32")
    Output
      PJRTBuffer 
       100000
      [ CPUf32{} ] 

---

    Code
      pjrt_scalar(-10^5, "f32")
    Output
      PJRTBuffer 
       -100000
      [ CPUf32{} ] 

---

    Code
      pjrt_scalar(250L, "ui8")
    Output
      PJRTBuffer 
       250
      [ CPUui8{} ] 

---

    Code
      pjrt_scalar(12L, "ui16")
    Output
      PJRTBuffer 
       12
      [ CPUui16{} ] 

---

    Code
      pjrt_scalar(0L, "ui32")
    Output
      PJRTBuffer 
       0
      [ CPUui32{} ] 

---

    Code
      pjrt_scalar(998L, "ui64")
    Output
      PJRTBuffer 
       998
      [ CPUui64{} ] 

---

    Code
      pjrt_scalar(14L, "i8")
    Output
      PJRTBuffer 
       14
      [ CPUi8{} ] 

---

    Code
      pjrt_scalar(-12L, "i16")
    Output
      PJRTBuffer 
       -12
      [ CPUi16{} ] 

---

    Code
      pjrt_scalar(0L, "i32")
    Output
      PJRTBuffer 
       0
      [ CPUi32{} ] 

---

    Code
      pjrt_scalar(998L, "i64")
    Output
      PJRTBuffer 
       998
      [ CPUi64{} ] 

---

    Code
      pjrt_scalar(TRUE)
    Output
      PJRTBuffer 
       1
      [ CPUpred{} ] 

---

    Code
      pjrt_scalar(FALSE)
    Output
      PJRTBuffer 
       0
      [ CPUpred{} ] 

# printer options

    Code
      print(pjrt_buffer(1:100), max_rows = 10)
    Output
      PJRTBuffer 
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
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUi32{100} ] 

---

    Code
      print(pjrt_buffer(1:100), max_rows = 100)
    Output
      PJRTBuffer 
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
        31
        32
        33
        34
        35
        36
        37
        38
        39
        40
        41
        42
        43
        44
        45
        46
        47
        48
        49
        50
        51
        52
        53
        54
        55
        56
        57
        58
        59
        60
        61
        62
        63
        64
        65
        66
        67
        68
        69
        70
        71
        72
        73
        74
        75
        76
        77
        78
        79
        80
        81
        82
        83
        84
        85
        86
        87
        88
        89
        90
        91
        92
        93
        94
        95
        96
        97
        98
        99
       100
      [ CPUi32{100} ] 

---

    Code
      print(pjrt_buffer(1:10000))
    Output
      PJRTBuffer 
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
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUi32{10000} ] 

---

    Code
      print(pjrt_buffer(1:11, shape = c(11, 1)), max_rows = 10)
    Output
      PJRTBuffer 
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
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUi32{11x1} ] 

---

    Code
      print(pjrt_buffer(c(100L), shape = c(1, 1)), max_width = 3, max_rows = 1)
    Output
      PJRTBuffer 
       100
      [ CPUi32{1x1} ] 

---

    Code
      print(x, max_width = 7, max_rows = 1)
    Output
      PJRTBuffer 
      Columns 1 to 1
       100
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUi32{1x3} ] 

---

    Code
      print(x, max_width = 8, max_rows = 1)
    Output
      PJRTBuffer 
      Columns 1 to 2
       100 100
       ... [output was truncated, set max_rows = -1 to see all]
      [ CPUi32{1x3} ] 

---

    Code
      print(pjrt_buffer(rep(1e-07, 50), shape = c(1, 50)), max_width = -1)
    Output
      PJRTBuffer 
      1e+07 *
       1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000
      [ CPUf32{1x50} ] 

# scale prefix is printed per slice

    Code
      pjrt_buffer(x, shape = c(2, 1, 2))
    Output
      PJRTBuffer 
      (1,.,.) =
      1e+07 *
       10.0000 10.0000
      
      (2,.,.) =
      1e+10 *
       10.0000 10.0000
      [ CPUf32{2x1x2} ] 

---

    Code
      print(pjrt_buffer(rep(x, 5), shape = c(2, 2, 5)), max_width = 15)
    Output
      PJRTBuffer 
      (1,.,.) =
      1e+07 *
      Columns 1 to 1
       10.0000
       10.0000
      Columns 2 to 2
       10.0000
       10.0000
      Columns 3 to 3
       10.0000
       10.0000
      Columns 4 to 4
       10.0000
       10.0000
      Columns 5 to 5
       10.0000
       10.0000
      
      (2,.,.) =
      1e+10 *
      Columns 1 to 1
       10.0000
       10.0000
      Columns 2 to 2
       10.0000
       10.0000
      Columns 3 to 3
       10.0000
       10.0000
      Columns 4 to 4
       10.0000
       10.0000
      Columns 5 to 5
       10.0000
       10.0000
      [ CPUf32{2x2x5} ] 

# metal

    Code
      pjrt_buffer(1:10, "f32", device = "metal")
    Output
      PJRTBuffer 
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
      [ METALf32{10} ] 

