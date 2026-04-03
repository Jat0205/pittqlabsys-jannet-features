'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 2
' Initial_Processdelay           = 3000
' Eventsource                    = External
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.3.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = SINGLENV-PC-1  SINGLENV-PC-1\Duttlab
'<Header End>
'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 2
' Initial_Processdelay           = 3000
' Eventsource                    = External
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.3.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = DUTTLAB8  Duttlab8\Duttlab
'<Header End>
'
' ODMR Pulsed Counter Script
' This is using external event trigger
' Using A, CLK 1 (input from APD) and external event trigger (input from AWG marker)
' external event is rising edge goto event
' When CLK rising edge: inc count, else: IDLE
' This file follows the old approach (awg triggers adwin to count)
' Later we can test awg520_trigger_test.bas and see if that is better (adwin triggers awg to move to next task/line)
DIM signal_count, total_count AS LONG
DIM count_time, reset_time, iteration_number as LONG
DIM Data_1[100000] AS LONG
DIM Data_2[100000] AS LONG
#Include ADwinGoldII.inc
init:
  Cnt_Enable(0)
  Cnt_Mode(1,8)          ' Counter 1 set to increasing
  Par_7 = 0   ' acquisition done flag
  Cnt_Clear(1)           ' Clear counter 1
  Par_8=0
  signal_count=0
  total_count=0
  iteration_number = 0
  ' NOTE: The offsets (10 and 30) are historical calibration values
  ' that were determined empirically. The actual timing values are:
  ' count_time = (Par_3-10)/10  where Par_3 is passed from Python
  ' reset_time = (Par_4-30)/10  where Par_4 is passed from Python
  ' These offsets ensure proper timing calibration for the hardware setup.
  count_time = (Par_3-10)/10 'added on 2/6/20 to allow passing parameter from Python
  reset_time = (Par_4-30)/10  'added on 2/6/20 to allow passing parameter from Python
event:
  Par_8=Par_8+1
  Cnt_Enable(1)          ' enable counter 1
  IO_Sleep(count_time)  ' count time 300 ns
  Cnt_Enable(0)          ' disable counter 1
  Cnt_Latch(1)           ' Latch counter 1
  IO_Sleep(reset_time)  ' reset time 1750 us
  Cnt_Enable(1)         ' enable counter 1
  IO_Sleep(count_time)  ' count time 300 ns
  Cnt_Enable(0)          ' disable counter 1
  signal_count=Cnt_Read_Latch(1)    ' accumulate signal counts
  total_count=Cnt_Read(1)           ' accumulate total counts (signal + reference)
  Cnt_Clear(1)           ' Clear counter 1
  iteration_number = iteration_number + 1 ' increase the iteration number
  Data_1[iteration_number] = Data_1[iteration_number] + signal_count
  Data_2[iteration_number] = Data_2[iteration_number] + total_count
  IF (iteration_number = Par_6) THEN 'if we did all of our iterations for a given sequence, then go back to iteration 0
    iteration_number = 0
  ENDIF
  ' Check if we've completed all repetitions and if iteration_number is 0 which means that it got to Par_6
  ' Par_5 contains the number of repetitions per scan point (e.g., 50000)
  IF ((Par_8>=Par_5) and (iteration_number = 0)) THEN
    Par_7=1
    Cnt_Enable(0)
    END
  ENDIF
