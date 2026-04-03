'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 3000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.3.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = SINGLENV-PC-1  SINGLENV-PC-1\Duttlab
'<Header End>
#Include ADwinGoldII.inc
DIM count_time AS FLOAT
DIM Data_1[1] AS LONG
'DIM trigger_duration AS FLOAT

init:
  Cnt_Enable(0)
  Cnt_Mode(1,8)   ' Counter 1 set to increasing
  'trigger_duration = 1  ' 10ns trigger pulse width (in 10 ns)

  Cnt_Clear(1)          ' Clear counter 1
  Processdelay = 150000000*3
  Data_1[1] =0
event:
  Data_1[1] = 0
  Cnt_Clear(1)
  Cnt_enable(1)
  CPU_Sleep(117404000)
  Data_1[1] = Cnt_read(1)
  Cnt_enable(0)
