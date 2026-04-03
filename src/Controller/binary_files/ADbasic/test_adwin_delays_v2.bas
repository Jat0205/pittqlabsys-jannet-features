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
'DIM trigger_duration AS FLOAT
' test 1
'DIGOUT(21, 1)
'DIGOUT(16, 1)
'CPU_Sleep(count_time)
'DIGOUT(21, 0)
'DIGOUT(16, 0)
'CPU_Sleep(count_time)
  
' test 2
'DIGOUT(21, 1)
'Cnt_enable(1)
'DIGOUT(16, 1)
'CPU_Sleep(count_time)
'DIGOUT(21, 0)
'DIGOUT(16, 0)
'CPU_Sleep(count_time)
  
' test 3
'DIGOUT(21, 1)
'Cnt_enable(0)
'DIGOUT(16, 1)
'CPU_Sleep(count_time)
'DIGOUT(21, 0)
'DIGOUT(16, 0)
'CPU_Sleep(count_time)
'Cnt_enable(1)

' test 4
'DIGOUT(21, 1)
'CPU_Sleep(count_time)
'DIGOUT(16, 1)
'DIGOUT(21, 0)
'DIGOUT(16, 0)

' test 5
'DIGOUT(21, 1)
'Cnt_read(count_time)
'DIGOUT(16, 1)
'DIGOUT(21, 0)
'DIGOUT(16, 0)

'test 6
'DIGOUT(21, 1)
'Cnt_Clear(1)
'DIGOUT(16, 1)
'DIGOUT(21, 0)
'DIGOUT(16, 0)

init:
  Cnt_Enable(1)
  Cnt_Mode(1,8)   ' Counter 1 set to increasing
  'trigger_duration = 1  ' 10ns trigger pulse width (in 10 ns)

  Cnt_Clear(1)          ' Clear counter 1
  count_time = 30 ' 300/10 as it's 300 ns but units are in 10 ns
  Conf_DIO(1100b) ' configure 0 - 15 as DIGIN, and 16 - 31 as DIGOUT
  ' Set digital output 21 to low (no trigger)
  DIGOUT(21, 0)
  DIGOUT(16, 0)
  Processdelay = 2*count_time*3
event:
  ' test 1
  DIGOUT(21, 1)
  CPU_Sleep(count_time)
  DIGOUT(21, 0)
  DIGOUT(16, 1)
  CPU_Sleep(count_time)
  DIGOUT(16, 0)
