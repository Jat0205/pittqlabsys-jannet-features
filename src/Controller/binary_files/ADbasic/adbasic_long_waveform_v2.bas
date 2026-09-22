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
' Process_Number               = 2
' Initial_Processdelay         = 3000
' Eventsource                  = External
' Control_long_Delays_for_Stop   = No
' Priority                     = High
' Version                      = 1
' ADbasic_Version              = 6.3.0
' Optimize                     = Yes
' Optimize_Level               = 1
' Stacksize                    = 1000
' Info_Last_Save               = DUTTLAB8  Duttlab8\Duttlab
'<Header End>
'
' ODMR Pulsed Counter Script: 2 pulses from proteus
' This is using external event trigger
' Using A, CLK 1 (input from APD) and external event trigger (input from AWG marker)
' external event is rising edge goto event
' When CLK rising edge: inc count, else: IDLE
' This file follows the old approach (awg triggers adwin to count)
#Include ADwinGoldII.inc
DIM signal_count, ref_count, number_of_signal_events AS LONG
DIM count_time, reset_time, iteration_number, event_id, i as LONG
DIM Data_1[20] AS LONG
DIM Data_2[20] AS LONG
init:
  event_id = 0   ' 1 for signal count and 2 for reference count
  Cnt_Enable(0)
  Cnt_Mode(1,8)        ' Counter 1 set to increasing
  Par_7 = 0       ' acquisition done flag
  Par_8 = 0    ' repetition_counter
  number_of_signal_events = Par_5 * Par_6
  Cnt_Clear(1)         ' Clear counter 1
  signal_count=0
  ref_count=0
  iteration_number = 0
  ' NOTE: The offsets (10 and 30) are historical calibration values
  ' that were determined empirically. The actual timing values are:
  ' count_time = (Par_3-10)/10  where Par_3 is passed from Python
  ' reset_time = (Par_4-30)/10  where Par_4 is passed from Python
  ' These offsets ensure proper timing calibration for the hardware setup.
  count_time = (Par_3-10)/10 'added on 2/6/20 to allow passing parameter from Python
  reset_time = (Par_4-30)/10  'added on 2/6/20 to allow passing parameter from Python
  i = 1
  DO
    Data_1[i] = 0
    Data_2[i] = 0
    i = i +1
  UNTIL (i = 21)
event:
  Cnt_Enable(0)
  Cnt_Clear(1)
  event_id = event_id + 1
  IF (event_id = 1) THEN
    Par_8 = Par_8 + 1                         ' current event number (only increase for signal count)
    Cnt_Enable(1)            ' enable counter 1
    IO_Sleep(count_time)      ' count time 300 ns
    Cnt_Enable(0)            ' disable counter 1
    signal_count=Cnt_Read(1)    ' accumulate signal counts
    Cnt_Clear(1)
    Data_1[iteration_number] = Data_1[iteration_number] + signal_count
  ENDIF
  ' This is done inside proteus: IO_Sleep(reset_time)   ' reset time 1750 us
  IF (event_id = 2) THEN
    Cnt_Enable(1)              ' enable counter 1
    IO_Sleep(count_time)  ' count time 300 ns
    Cnt_Enable(0)        ' disable counter 1
    ref_count=Cnt_Read(1)         ' accumulate reference counts
    Cnt_Clear(1)             ' Clear counter 1
    Data_2[iteration_number] = Data_2[iteration_number] + ref_count
    iteration_number = iteration_number + 1 ' increase the iteration number (only increase for ref counts)
  ENDIF
  signal_count = 0
  ref_count = 0
  IF (iteration_number = Par_6) THEN 'if we did all of our iterations for a given sequence, then go back to iteration 0
    iteration_number = 0
  ENDIF
  IF (event_id = 2) THEN
    event_id = 0
  ENDIF
  ' Check if we've completed all repetitions and if iteration_number is 0 which means that it got to Par_6
  ' Par_5 contains the number of repetitions per scan point (e.g., 50000)
  IF (Par_8 >= number_of_signal_events) THEN
    Par_7=1
    Cnt_Enable(0)
    END
  ENDIF
