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
  Processdelay =  ' in 10/3 for T11 processor
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