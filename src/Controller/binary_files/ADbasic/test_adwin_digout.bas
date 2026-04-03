'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 1000
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
' This process generates trigger pulses to control proteus external triggering
' for testing the ADwin -> proteus control architecture.
'
' Hardware Setup:
' - ADwin Digital Output -> Proteus TRIG 1 IN (front panel)
' - Proteus configured for external trigger with Wait Trigger enabled
' - Computer controls JUMP_MODE software for sequence advancement
'
' Operation:
' - Process generates trigger pulses at specified intervals
' - Each trigger causes Proteus to advance to next sequence line
' - Computer can control timing and number of triggers via parameters
' This is the new approach: adwin triggers awg to move to next task/line

#Include ADwinGoldII.inc
DIM trigger_counter AS LONG
DIM trigger_interval AS LONG
DIM trigger_duration AS LONG
DIM current_state AS LONG


init:
  ' Initialize variables
  trigger_counter = 0
  trigger_interval = 100000000    ' 1 second default (in 10ns)
  trigger_duration = 1000       ' 10us trigger pulse width (in 10 ns)
  current_state = 0             ' 0=idle, 1=trigger_high, 2=trigger_low
  Conf_DIO(1100b)

  ' Set digital output 21 to low (no trigger)
  DIGOUT(21, 0)

  ' Set parameters for Python control
  Par_1 = trigger_counter       ' Current trigger count
  Par_2 = trigger_interval      ' Trigger interval in microseconds
  Par_4 = trigger_duration      ' Trigger pulse duration in microseconds
  Par_5 = current_state         ' Current state (0=idle, 1=high, 2=low)


event:
  ' Main trigger generation loop
  DO
    IF (current_state = 0) THEN
      ' Idle state - check if we should generate a trigger
      DIGOUT(21, 1)              ' Set trigger high
      current_state = 1
      Par_5 = current_state


    ELSE
      IF (current_state = 1) THEN
        ' Trigger high state - wait for pulse duration
        CPU_Sleep(trigger_duration)
        DIGOUT(21, 0)                ' Set trigger low
        current_state = 2
        Par_5 = current_state

      ELSE
        IF (current_state = 2) THEN
          ' Trigger low state - wait for interval before next trigger
          CPU_Sleep(trigger_interval - trigger_duration)
          trigger_counter = trigger_counter + 1
          Par_1 = trigger_counter
          current_state = 0            ' Back to idle
          Par_5 = current_state
        ENDIF
      ENDIF
    ENDIF
  UNTIL (Par_6 = 1)
  ' Clean up
  DIGOUT(21, 0)                  ' Ensure trigger is low
  current_state = 0
  Par_5 = current_state
  END
