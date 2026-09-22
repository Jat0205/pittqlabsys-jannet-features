"""
Sequence Builder Module

This module converts SequenceDescription objects into optimized Sequence objects
that can be processed by hardware-specific optimizers.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import numpy as np

from .sequence_description import SequenceDescription, PulseDescription, LoopDescription, ConditionalDescription, MarkerDescription
from .pulses import (Pulse, GaussianPulse, SechPulse, LorentzianPulse, SquarePulse,
                     DataPulse, MarkerEvent, ChirpPulse, HyperbolicSecantPulse,
                     AsymmetricHyperbolicSecantPulse)
from .sequence import Sequence

def _display_envelope(pulse) -> np.ndarray:
    """Amplitude envelope for PLOTTING only.

    generate_samples() on a frequency-swept pulse (chirp / hs / asymm_hs) returns
    one oscillating IQ quadrature (A(t)cos phi or A(t)sin phi). Drawn across a full
    sequence axis those ~ns-period oscillations pack together and read as a solid
    rectangle. For those pulses, reconstruct the true amplitude A(t) = sqrt(I^2+Q^2)
    so the envelope shows. Non-swept pulses are returned unchanged.
    """
    env = np.asarray(pulse.generate_samples(), dtype=float)
    if hasattr(pulse, "bandwidth") and hasattr(pulse, "quadrature"):
        saved = pulse.quadrature
        try:
            pulse.quadrature = "I"; I = np.asarray(pulse.generate_samples(), dtype=float)
            pulse.quadrature = "Q"; Q = np.asarray(pulse.generate_samples(), dtype=float)
        finally:
            pulse.quadrature = saved
        if I.size and I.size == Q.size:
            return np.sqrt(I ** 2 + Q ** 2)
    return env


class SequenceBuilder:
    """
    Converts SequenceDescription objects to optimized Sequence objects.
    
    This class handles:
    - Building sequences from descriptions
    - Generic memory optimization (hardware-agnostic)
    - Splitting long sequences into chunks
    - Handling loops and conditionals
    """
    
    def __init__(self, sample_rate: float = 1e9):
        """
        Initialize the sequence builder.
        
        Args:
            sample_rate: Default sample rate in Hz
        """
        self.sample_rate = sample_rate
        # Note: Memory constraints are handled by hardware-specific optimizers
        # This class is hardware-agnostic
    
    def build_sequence(self, description: SequenceDescription) -> OptimizedSequence:
        """
        Build an optimized sequence from a sequence description.
        
        Args:
            description: SequenceDescription object
            
        Returns:
            OptimizedSequence object ready for hardware processing
            
        Raises:
            ValueError: If description is invalid
            BuildError: If sequence building fails
        """
        try:
            # Validate the description first
            if not description.validate():
                raise BuildError("Sequence description validation failed")
            # Calculate total sequence length in samples
            total_samples = int(description.total_duration * self.sample_rate)
            # Create the main sequence
            main_sequence = Sequence(total_samples)
            # Add all pulses
            for pulse_desc in description.pulses:
                pulse_obj = self._create_pulse_object(pulse_desc)
                start_sample = int(pulse_desc.timing * self.sample_rate)
                main_sequence.add_pulse(start_sample, pulse_obj)
            for marker in description.markers:
                start_sample = int(marker.start_time * self.sample_rate)
                end_sample = start_sample + int(marker.duration * self.sample_rate)
                mk_event = MarkerEvent(
                    name=f"{marker.name}_{marker.channel}",
                    length=main_sequence.length,  # must match Sequence length
                    on_index=start_sample,
                    off_index=end_sample
                )
                main_sequence.add_marker(mk_event)


            # Handle loops
            """for loop_desc in description.loops:
                loop_sequence = self._build_loop_sequence(loop_desc)
                # For now, we'll add loop sequences as separate sequences
                # In a more sophisticated implementation, we'd handle repetition
            print(f"Main sequence after adding loops: {main_sequence}")"""
            # Handle conditionals
            """for conditional_desc in description.conditionals:
                conditional_sequence = self._build_conditional_sequence(conditional_desc)
                # For now, we'll add conditional sequences as separate sequences
                # In a more sophisticated implementation, we'd handle branching"""
            
            # Create optimized sequence (single chunk for now)
            optimized = OptimizedSequence(
                name=description.name,
                sequences=[main_sequence],
                metadata={
                    "experiment_type": description.experiment_type,
                    "sample_rate": self.sample_rate,
                    "total_duration": description.total_duration,
                    "variables": description.variables
                }
            )

            return optimized
            
        except Exception as e:
            if isinstance(e, BuildError):
                raise
            raise BuildError(f"Failed to build sequence: {e}")
    
    def build_from_preset(self, preset_name: str, **parameters) -> OptimizedSequence:
        """
        Build a sequence from a preset experiment.
        
        Args:
            preset_name: Name of the preset experiment
            **parameters: Custom parameters for the preset
            
        Returns:
            OptimizedSequence object
            
        Raises:
            ValueError: If preset not found
        """
        # This would integrate with the sequence parser to get preset descriptions
        # For now, we'll raise NotImplementedError
        raise NotImplementedError("Preset integration not yet implemented")
    
    def optimize_for_memory_constraints(self, sequence: Sequence, max_samples_per_chunk: int) -> List[Sequence]:
        try:
            # Sequence has no materialized buffer; its sample count is `length`
            if sequence.length <= max_samples_per_chunk:
                return [sequence]


            split_points = self._find_optimal_split_points(sequence, max_samples_per_chunk)
            return self._split_sequence_at_boundaries(sequence, max_samples_per_chunk)


        except Exception as e:
            raise OptimizationError(f"Failed to optimize sequence: {e}")

    def build_scan_sequences(self, description: SequenceDescription) -> List[Sequence]:
        """Build multiple sequences for each scan point with proper timing adjustments.
        
        This method handles variable scanning by:
        1. Generating all variable value combinations
        2. Creating a sequence for each combination
        3. Adjusting timing of subsequent pulses based on variable changes
        4. Respecting [fixed] markers that prevent timing adjustment
        
        Args:
            description: SequenceDescription with variables to scan
            
        Returns:
            List of Sequence objects, one for each scan point
            
        Raises:
            BuildError: If sequence building fails"""

        pulses = description.pulses  # pulses outside loops
        variable_loops = description.loops  # loops that contain variable pulses

        markers = description.markers
        try:
            if not description.variables:
                # No variables to scan, return single sequence
                optimized_sequence = self.build_sequence(description)
                self.sample_rate = optimized_sequence.metadata["sample_rate"]
                main_sequence = optimized_sequence.sequences[0]
                main_sequence.name = f"{description.name}_scan"
                return [main_sequence]

                # Validate single variable scanning
            if len(description.variables) > 1:
                import warnings
                warnings.warn(
                    f"Building {len(description.variables)} variables simultaneously. "
                    f"This creates {self._calculate_total_combinations(description.variables)} sequences. "
                    "Consider scanning one variable at a time for clean data correlation.",
                    UserWarning
                )

            # Generate all variable value combinations
            variable_combinations = self._generate_variable_combinations(description.variables)
            scan_sequences = []
            for combo in variable_combinations:
                # Create sequence with these variable values
                optimized_sequence = self._create_sequence_with_variables(description, combo)
                # Get the main sequence from the optimized sequence
                main_sequence = optimized_sequence.sequences[0]

                # Apply timing adjustments for scanned variables
                """main_sequence = self._adjust_timing_for_variable_scan(main_sequence, combo)"""
                # Recalculate actual duration based on adjusted timing
                actual_duration = self._calculate_actual_duration_from_sequence(main_sequence)
                main_sequence.total_duration = actual_duration

                scan_sequences.append(main_sequence)
            return scan_sequences

        except Exception as e:
            raise BuildError(f"Failed to build scan sequences: {e}")
    
    def _generate_variable_combinations(self, variables: Dict[str, VariableDescription]) -> List[Dict[str, float]]:
        """Generate all combinations of variable values."""
        if not variables:
            return [{}]
        
        # Get the first variable
        var_name = list(variables.keys())[0]
        var_desc = variables[var_name]
        
        # For single variable scanning, just create list of values
        combinations = []
        for value in var_desc.values:
            combinations.append({var_name: value})

        return combinations
    
    def _create_sequence_with_variables(self, description: SequenceDescription, variable_values: Dict[str, float]) -> OptimizedSequence:
        """Create a sequence with specific variable values substituted."""
        # Create a copy of the description with variable values
        modified_description = SequenceDescription(
            name=f"{description.name}_scan",
            experiment_type=description.experiment_type,
            total_duration=description.total_duration,
            sample_rate=description.sample_rate,
            repeat_count=description.repeat_count
        )
        # Add pulses with variable values substituted
        sequence_duration = 0.0
        for pulse in description.pulses:
            modified_pulse = PulseDescription(
                name=pulse.name,
                pulse_type=pulse.pulse_type,
                channel=pulse.channel,
                shape=pulse.shape,
                duration=pulse.duration,
                amplitude=pulse.amplitude,
                timing=pulse.timing,
                timing_type=pulse.timing_type,
                parameters=pulse.parameters.copy(),
                markers=pulse.markers.copy(),
                fixed_timing=pulse.fixed_timing
            )
            # Substitute variable values in parameters
            for param_name, param_value in modified_pulse.parameters.items():
                if isinstance(param_value, str) and param_value in variable_values:
                    modified_pulse.parameters[param_name] = variable_values[param_value]

            # apply variable substitutions
            modified_pulse.duration = self._evaluate_expression(pulse.duration, variable_values)
            modified_pulse.timing = self._evaluate_expression(pulse.timing, variable_values)
            modified_pulse.amplitude = self._evaluate_expression(pulse.amplitude, variable_values)
            end_time = modified_pulse.timing + modified_pulse.duration
            sequence_duration = max(sequence_duration, end_time)
            modified_description.add_pulse(modified_pulse)

        modified_description.total_duration = sequence_duration
        for marker in description.markers:
            modified_marker = MarkerDescription(
                name=marker.name,
                channel=marker.channel,
                start_time=marker.start_time,
                duration=marker.duration,
                state=marker.state
            )

            # Substitute variable values in duration if it's a variable
            modified_marker.duration = self._evaluate_expression(marker.duration, variable_values)
            modified_marker.start_time = self._evaluate_expression(marker.start_time, variable_values)
            end_time = modified_marker.start_time + modified_marker.duration

            sequence_duration = max(sequence_duration, end_time)
            modified_description.add_marker(modified_marker)
        modified_description.total_duration = sequence_duration

        # Add other components
        for loop in description.loops:
            modified_description.add_loop(loop)
        """for conditional in description.conditionals:
            modified_description.add_conditional(conditional)"""
        
        return self.build_sequence(modified_description)

    def _evaluate_expression(self, expr, variables) -> float:
        import ast
        import operator
        """
        Safely evaluate a numeric expression with variables.

        Examples:
            "pulse_duration"
            "2 * pulse_duration"
            "3 * pulse_duration + 100e-9"
        """
        # Case 1: already numeric
        if isinstance(expr, (int, float)):
            return float(expr)

        # Case 2: must be string
        if not isinstance(expr, str):
            raise TypeError(f"Invalid expression type: {type(expr)}")

        expr = expr.strip()
        # Allowed operators
        import math
        # Allowed operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        # Allowed functions and constants (e.g. phase sweeps: cos(phase), sin(phase))
        functions = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
            "abs": abs,
        }
        constants = {"pi": math.pi, "e": math.e}

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)

            elif isinstance(node, ast.Num):  # Python <3.8
                return node.n

            elif isinstance(node, ast.Constant):  # Python 3.8+
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Invalid constant")

            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](
                    _eval(node.left),
                    _eval(node.right)
                )

            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](_eval(node.operand))

            elif isinstance(node, ast.Call):
                # function call, e.g. cos(phase), sin(phase/2)
                if not isinstance(node.func, ast.Name) or node.func.id not in functions:
                    raise ValueError(f"Unsupported function in expression: {expr}")
                if node.keywords:
                    raise ValueError(f"Keyword arguments not allowed: {expr}")
                return functions[node.func.id](*[_eval(arg) for arg in node.args])

            elif isinstance(node, ast.Name):
                # variables take precedence, then math constants (pi, e)
                if node.id in variables:
                    return variables[node.id]
                if node.id in constants:
                    return constants[node.id]
                raise KeyError(f"Unknown variable '{node.id}'")

            else:
                raise ValueError(f"Unsupported expression: {expr}")

        tree = ast.parse(expr, mode="eval")
        return float(_eval(tree))

    def _adjust_timing_for_variable_scan(self, sequence: Sequence, variable_values: Dict[str, float]) -> Sequence:
        """Adjust timing of pulses based on variable changes, respecting [fixed] markers."""
        if not variable_values:
            return sequence
        
        # Find the scanned variable (we assume single variable scanning)
        var_name = list(variable_values.keys())[0]
        var_value = variable_values[var_name]
        
        # For now, assume the first pulse uses the scanned variable
        # In a more sophisticated implementation, we'd check if the pulse duration
        # actually matches the variable name or description
        if not sequence.pulses:
            return sequence
        
        # Sort pulses by start time
        sorted_pulses = sorted(sequence.pulses, key=lambda x: x[0])
        
        # Find the first pulse (that uses the scanned variable)
        first_start_sample, first_pulse = sorted_pulses[0]
        
        # Convert current pulse length to duration
        current_duration = first_pulse.length / self.sample_rate
        
        # Calculate how much the timing needs to shift
        duration_change = var_value - current_duration
        
        if duration_change == 0:
            return sequence
        
        # Update the first pulse length based on new duration
        new_length = int(var_value * self.sample_rate)
        first_pulse.length = new_length
        
        # Move ALL subsequent pulses by the duration change
        # UNLESS they are marked as [fixed]
        for start_sample, pulse in sorted_pulses[1:]:  # Skip the first pulse
            if not getattr(pulse, 'fixed_timing', False):
                # Calculate new start sample
                new_start_sample = start_sample + int(duration_change * self.sample_rate)
                # Update the pulse timing
                sequence.pulses.remove((start_sample, pulse))
                sequence.pulses.append((new_start_sample, pulse))
            # If pulse has fixed_timing=True, leave it at its original position
        
        return sequence
    
    def _calculate_actual_duration(self, sequence: OptimizedSequence) -> float:
        """Calculate the actual duration of a sequence based on pulse timing."""
        if not sequence.sequences:
            return 0.0
        
        main_sequence = sequence.sequences[0]
        if not main_sequence.pulses:
            return 0.0
        
        max_end_time = 0.0
        for start_sample, pulse in main_sequence.pulses:
            pulse_end_time = start_sample / self.sample_rate + pulse.length / self.sample_rate
            max_end_time = max(max_end_time, pulse_end_time)
        
        return max_end_time
    
    def _calculate_actual_duration_from_sequence(self, sequence: Sequence) -> float:
        """Calculate the actual duration of a sequence based on pulse timing."""
        if not sequence.pulses:
            return 0.0
        max_end_time = 0.0
        for start_sample, pulse in sequence.pulses:

            pulse_end_time = start_sample / self.sample_rate + pulse.length / self.sample_rate
            max_end_time = max(max_end_time, pulse_end_time)

        return max_end_time
    
    def _calculate_total_combinations(self, variables: Dict[str, VariableDescription]) -> int:
        """Calculate total number of scan combinations."""
        total = 1
        for var_desc in variables.values():
            total *= var_desc.steps
        return total
    
    def _create_pulse_object(self, pulse_desc: PulseDescription) -> Pulse:
        """
        Create a Pulse object from a PulseDescription.
        
        Args:
            pulse_desc: Pulse description
            
        Returns:
            Pulse object of appropriate type
        """
        # Calculate pulse length in samples
        pulse_length = int(pulse_desc.duration * self.sample_rate)

        # Create pulse based on shape
        if pulse_desc.shape.value == "gaussian":
            # For Gaussian, sigma controls the width
            # Use duration/6 as sigma to get reasonable shape
            sigma = pulse_length / 6.0
            return GaussianPulse(
                name=pulse_desc.name,
                length=pulse_length,
                sigma=sigma,
                amplitude=pulse_desc.amplitude,
                fixed_timing=getattr(pulse_desc, 'fixed_timing', False)
            )
        
        elif pulse_desc.shape.value == "sech":
            # For Sech, width controls the shape
            width = pulse_length / 4.0
            return SechPulse(
                name=pulse_desc.name,
                length=pulse_length,
                width=width,
                amplitude=pulse_desc.amplitude,
                fixed_timing=getattr(pulse_desc, 'fixed_timing', False)
            )
        
        elif pulse_desc.shape.value == "lorentzian":
            # For Lorentzian, gamma controls the width
            gamma = pulse_length / 4.0
            return LorentzianPulse(
                name=pulse_desc.name,
                length=pulse_length,
                gamma=gamma,
                amplitude=pulse_desc.amplitude,
                fixed_timing=getattr(pulse_desc, 'fixed_timing', False)
            )
        
        elif pulse_desc.shape.value == "square":
            return SquarePulse(
                name=pulse_desc.name,
                length=pulse_length,
                amplitude=pulse_desc.amplitude,
                fixed_timing=getattr(pulse_desc, 'fixed_timing', False)
            )
        
        elif pulse_desc.shape.value == "sine":
            # For sine, we need frequency - use a reasonable default
            # This could be enhanced with actual frequency parameters
            return SquarePulse(  # Fallback to square for now
                name=pulse_desc.name,
                length=pulse_length,
                amplitude=pulse_desc.amplitude,
                fixed_timing=getattr(pulse_desc, 'fixed_timing', False)
            )
        
        elif pulse_desc.shape.value == "loadfile":
            # For loadfile, we'd need to implement file loading
            # For now, fallback to square pulse
            return SquarePulse(
                name=pulse_desc.name,
                length=pulse_length,
                amplitude=pulse_desc.amplitude
            )
        
        elif pulse_desc.shape.value == "data":
            # For data pulses, we need a filename parameter
            filename = pulse_desc.get_parameter("filename")
            if not filename:
                raise ValueError(f"DataPulse requires 'filename' parameter for {pulse_desc.name}")
            
            return DataPulse(
                name=pulse_desc.name,
                length=pulse_length,
                filename=filename
            )
        elif pulse_desc.shape.value == "chirp":
            bandwidth = pulse_desc.get_parameter("bandwidth")
            if bandwidth is None:
                raise ValueError(f"chirp pulse '{pulse_desc.name}' requires a 'bandwidth' parameter (Hz)")
            return ChirpPulse(
                name=pulse_desc.name, length=pulse_length, sample_rate=self.sample_rate,
                bandwidth=float(bandwidth), amplitude=pulse_desc.amplitude,
                quadrature=pulse_desc.get_parameter("quadrature", "I"),
                center_freq=float(pulse_desc.get_parameter("center_freq", 0.0)),
                edge_fraction=float(pulse_desc.get_parameter("edge_fraction", 0.0)),
                phase0=float(pulse_desc.get_parameter("phase0", 0.0)),
                fixed_timing=getattr(pulse_desc, 'fixed_timing', False)
            )

        elif pulse_desc.shape.value == "hs":
            bandwidth = pulse_desc.get_parameter("bandwidth")
            if bandwidth is None:
                raise ValueError(f"hs pulse '{pulse_desc.name}' requires a 'bandwidth' parameter (Hz)")
            return HyperbolicSecantPulse(
                name=pulse_desc.name, length=pulse_length, sample_rate=self.sample_rate,
                bandwidth=float(bandwidth), beta=float(pulse_desc.get_parameter("beta", 5.3)),
                amplitude=pulse_desc.amplitude,
                quadrature=pulse_desc.get_parameter("quadrature", "I"),
                center_freq=float(pulse_desc.get_parameter("center_freq", 0.0)),
                phase0=float(pulse_desc.get_parameter("phase0", 0.0)),
                fixed_timing=getattr(pulse_desc, 'fixed_timing', False)
            )

        elif pulse_desc.shape.value == "asymm_hs":
            bandwidth = pulse_desc.get_parameter("bandwidth")
            if bandwidth is None:
                raise ValueError(f"asymm_hs pulse '{pulse_desc.name}' requires a 'bandwidth' parameter (Hz)")
            return AsymmetricHyperbolicSecantPulse(
                name=pulse_desc.name, length=pulse_length, sample_rate=self.sample_rate,
                bandwidth=float(bandwidth),
                n_left=float(pulse_desc.get_parameter("n_left", 1.0)),
                n_right=float(pulse_desc.get_parameter("n_right", 6.0)),
                beta=float(pulse_desc.get_parameter("beta", 5.3)),
                amplitude=pulse_desc.amplitude,
                quadrature=pulse_desc.get_parameter("quadrature", "I"),
                center_freq=float(pulse_desc.get_parameter("center_freq", 0.0)),
                phase0=float(pulse_desc.get_parameter("phase0", 0.0)),
                fixed_timing=getattr(pulse_desc, 'fixed_timing', False)
            )
        
        else:
            # Default to square pulse
            return SquarePulse(
                name=pulse_desc.name,
                length=pulse_length,
                amplitude=pulse_desc.amplitude
            )
    
    def _build_loop_sequence(self, loop_desc: LoopDescription) -> Sequence:
        """
        Build a sequence for a loop block.
        
        Args:
            loop_desc: Loop description
            
        Returns:
            Sequence object for the loop
        """
        # Calculate loop duration in samples
        loop_duration = loop_desc.end_time - loop_desc.start_time
        loop_samples = int(loop_duration * self.sample_rate)
        # Create sequence for the loop
        loop_sequence = Sequence(loop_samples)
        # Add pulses within the loop
        for pulse_desc in loop_desc.pulses:
            pulse_obj = self._create_pulse_object(pulse_desc)
            # Adjust timing relative to loop start
            relative_start = int((pulse_desc.timing - loop_desc.start_time) * self.sample_rate)
            if relative_start >= 0 and relative_start < loop_samples:
                loop_sequence.add_pulse(relative_start, pulse_obj)
        
        return loop_sequence
    
    def _build_conditional_sequence(self, conditional_desc: ConditionalDescription) -> Sequence:
        """
        Build a sequence for a conditional block.
        
        Args:
            conditional_desc: Conditional description
            
        Returns:
            Sequence object for the conditional
        """
        # Calculate conditional duration in samples
        conditional_duration = conditional_desc.end_time - conditional_desc.start_time
        conditional_samples = int(conditional_duration * self.sample_rate)
        
        # Create sequence for the conditional
        conditional_sequence = Sequence(conditional_samples)
        
        # Add true pulses (we'll handle branching logic later)
        for pulse_desc in conditional_desc.true_pulses:
            pulse_obj = self._create_pulse_object(pulse_desc)
            relative_start = int((pulse_desc.timing - conditional_desc.start_time) * self.sample_rate)
            if relative_start >= 0 and relative_start < conditional_samples:
                conditional_sequence.add_pulse(relative_start, pulse_obj)
        
        return conditional_sequence
    
    def _calculate_memory_usage(self, sequence: Sequence) -> int:
        """
        Calculate memory usage of a sequence in samples.
        
        Args:
            sequence: Sequence to analyze
            
        Returns:
            Number of samples required
        """
        if hasattr(sequence, 'waveform'):
            try:
                return len(sequence.waveform)
            except (TypeError, AttributeError):
                # If len() fails (e.g., on Mock objects), fall back to length attribute
                return getattr(sequence, 'length', 0)
        else:
            return getattr(sequence, 'length', 0)
    
    def _split_sequence_at_boundaries(self, sequence: Sequence, max_samples: int) -> List[Sequence]:
        total_samples = sequence.length
        if total_samples <= max_samples:
            return [sequence]


        chunks = []
        for start in range(0, total_samples, max_samples):
            end = min(start + max_samples, total_samples)
            chunk = Sequence(end - start)


            # re-home any pulse whose start index lands in this chunk
            for p_start, pulse in sequence.pulses:
                if start <= p_start < end:
                    chunk.add_pulse(p_start - start, pulse)


            # re-home markers by their on_index, rebuilt against the chunk length
            for mk in sequence.markers:
                if start <= mk.on_index < end:
                    chunk.add_marker(MarkerEvent(
                        name=mk.name,
                        length=chunk.length,
                        on_index=mk.on_index - start,
                        off_index=min(mk.off_index - start, chunk.length),
                    ))
            chunks.append(chunk)
        return chunks
    
    def _find_optimal_split_points(self, sequence: Sequence, max_samples: int) -> List[int]:
        total_samples = sequence.length
        if total_samples <= max_samples:
            return []
        return list(range(max_samples, total_samples, max_samples))

    def plot_sequence(self, sequence: Sequence, title: str = None, 
                     show_legend: bool = True, save_path: str = None) -> 'matplotlib.figure.Figure':
        """
        Create a static plot of a single sequence.
        
        Args:
            sequence: Sequence object to plot
            title: Optional title for the plot
            show_legend: Whether to show the legend
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
            
        Note: This method requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique channels from the sequence
        channels = set()
        for start_sample, pulse in sequence.pulses:
            if hasattr(pulse, 'channel'):
                channels.add(pulse.channel)
            else:
                # Default channel assignment based on pulse name
                if 'pi_2' in pulse.name.lower():
                    channels.add(1)
                elif 'laser' in pulse.name.lower():
                    channels.add(2)
                elif 'counter' in pulse.name.lower():
                    channels.add(3)
                elif 'trigger' in pulse.name.lower():
                    channels.add(5)
                else:
                    channels.add(1)
        
        channels = sorted(list(channels))
        
        # Channel colors and names
        colors = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple'}
        channel_names = {1: 'Pi/2', 2: 'Laser', 3: 'Counter', 4: 'Channel 4', 5: 'Trigger'}
        
        # Calculate dynamic x-axis limits
        max_time = 1000  # Default minimum
        for start_sample, pulse in sequence.pulses:
            pulse_end_time = start_sample / self.sample_rate * 1e9 + pulse.length / self.sample_rate * 1e9
            max_time = max(max_time, pulse_end_time + 100)  # Add 100ns padding
        
        # Set plot limits
        ax.set_xlim(0, max_time)
        ax.set_ylim(min(channels) - 0.5, max(channels) + 1.0)
        
        # Create time array
        time_ns = np.arange(0, int(max_time), 1)
        
        # Create signal arrays for each channel
        channel_signals = {}
        for channel in channels:
            channel_signals[channel] = np.zeros_like(time_ns, dtype=float)
        
        # Fill in the signals based on pulses
        for start_sample, pulse in sequence.pulses:
            start_time_ns = int(start_sample / self.sample_rate * 1e9)
            duration_ns = int(pulse.length / self.sample_rate * 1e9)
            end_time_ns = start_time_ns + duration_ns
            
            # Determine channel
            if hasattr(pulse, 'channel'):
                channel = pulse.channel
            else:
                if 'pi_2' in pulse.name.lower():
                    channel = 1
                elif 'laser' in pulse.name.lower():
                    channel = 2
                elif 'counter' in pulse.name.lower():
                    channel = 3
                elif 'trigger' in pulse.name.lower():
                    channel = 5
                else:
                    channel = 1
            
            # Create pulse shape based on pulse type
            if hasattr(pulse, 'generate_samples'):
                # Use the pulse's actual shape from SequenceBuilder
                try:
                    # Get the actual pulse envelope that was already generated
                    pulse_envelope = _display_envelope(pulse)
                    
                    # Scale and position the envelope for visualization
                    pulse_time = np.arange(max(0, start_time_ns), min(len(time_ns), end_time_ns))
                    if len(pulse_time) > 0:
                        # Map pulse envelope to time range
                        envelope_indices = np.linspace(0, len(pulse_envelope)-1, len(pulse_time), dtype=int)
                        envelope_values = pulse_envelope[envelope_indices]
                        
                        # Scale to visualization amplitude (0.4 above baseline)
                        scaled_values = 0.4 * envelope_values / np.max(np.abs(envelope_values))
                        
                        valid_indices = (pulse_time >= 0) & (pulse_time < len(time_ns))
                        channel_signals[channel][pulse_time[valid_indices]] = scaled_values[valid_indices]
                except Exception as e:
                    # If pulse generation fails, use square pulse as fallback
                    print(f"Warning1: Could not generate pulse shape for {pulse.name}, using square fallback: {e}")
                    if start_time_ns >= 0 and start_time_ns < len(time_ns):
                        end_idx = min(len(time_ns), end_time_ns)
                        channel_signals[channel][start_time_ns:end_idx] = 0.4
            else:
                # For pulses without generate_samples method, use square pulse
                if start_time_ns >= 0 and start_time_ns < len(time_ns):
                    end_idx = min(len(time_ns), end_time_ns)
                    channel_signals[channel][start_time_ns:end_idx] = 0.4
        
        # Plot signals
        for channel in channels:
            signal_y = channel + channel_signals[channel]
            ax.plot(time_ns, signal_y, color=colors.get(channel, 'black'), 
                   linewidth=2, alpha=0.8, label=channel_names.get(channel, f'Channel {channel}'))
        
        # Add labels and title
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Channel')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'Sequence: {getattr(sequence, "name", "Unnamed")}')
        
        # Set y-axis
        ax.set_yticks(channels)
        ax.set_yticklabels([channel_names.get(c, f'Channel {c}') for c in channels])
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend(loc='upper right')
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def animate_scan_sequences(self, sequences, fig, ax, title=None, interval=1000):
        """
        Create an animation showing the progression through scan sequences.

        Args:
            sequence: List of Sequence objects to animate
            title: Optional title for the animation
            interval: Animation interval in milliseconds

        Returns:
            matplotlib Animation object

        Note: This method requires matplotlib to be installed.
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            raise ImportError("matplotlib is required")

        # Get unique channels from all sequences
        all_channels = set()
        all_markers = set()
        for seq in sequences:
            for start_sample, pulse in seq.pulses:
                if hasattr(pulse, 'channel'):
                    all_channels.add(pulse.channel)
                else:
                    all_channels.add(int(pulse.name.split("_")[-1]))

            for marker in seq.markers:
                mk_ch = int(marker.name.split("_")[-1])
                all_markers.add(mk_ch)

        channels = sorted(list(all_channels))
        markers = sorted(list(all_markers))

        # Channel colors and names
        colors = {1: 'blue', 2: 'darkturquoise', 3: 'orange', 4: 'purple', 5: 'cyan'}
        channel_names = {1: 'ch 1', 2: 'ch 2', 3: 'ch 3', 4: 'ch 4', 5: 'Trigger'}
        mk_colors = {1: 'green', 2: 'red', 3: 'olive', 4: 'gray'}
        marker_names = {1: 'mkr 1', 2: 'mkr 2', 3: 'mkr 3', 4: 'mkr 4'}
        # Calculate global x-axis limits
        max_time = 1000
        for seq in sequences:
            for start_sample, pulse in seq.pulses:
                pulse_end_time = start_sample / self.sample_rate * 1e9 + pulse.length / self.sample_rate * 1e9
                max_time = max(max_time, pulse_end_time + 100)

        # Set plot limits
        ax.set_xlim(0, max_time)
        ax.set_ylim(min(channels) - 0.5, max(channels) + 1.0)

        # Create time array
        time_ns = np.arange(0, int(max_time), 1)

        def animate(frame):
            ax.clear()

            seq = sequences[frame]

            # Set labels and title
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Channel')
            if title:
                ax.set_title(f'{title} - Frame {frame + 1}/{len(sequences)}')
            else:
                ax.set_title(f'Sequence {frame + 1}/{len(sequences)}')
            marker_offset = -len(markers) - 1
            # Set plot limits
            ax.set_xlim(0, max_time)

            # Set y-axis
            #ax.set_yticks(channels)
            #ax.set_yticklabels([channel_names.get(c, f'Channel {c}') for c in channels])
            yticks = (
                    [marker_offset + m for m in markers] +
                    channels
            )
            yticklabels = (
                    [marker_names.get(m, f'Marker {m}') for m in markers] +
                    [channel_names.get(c, f'Channel {c}') for c in channels]
            )

            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)

            # Create signal arrays
            channel_signals = {}
            for channel in channels:
                channel_signals[channel] = np.zeros_like(time_ns, dtype=float)
            marker_signals = {}
            for marker in markers:
                marker_signals[marker] = np.zeros_like(time_ns, dtype=float)
            #markers:
            for mark in seq.markers:
                start_time_ns = int(mark.on_index / self.sample_rate * 1e9)
                end_time_ns = int(mark.off_index / self.sample_rate * 1e9)

                # Determine channel
                if hasattr(mark, 'channel'):
                    marker_channel = mark.channel
                else:
                    marker_channel = int(mark.name.split("_")[-1])

                if start_time_ns >= 0 and start_time_ns < len(time_ns):
                    end_idx = min(len(time_ns), end_time_ns)
                    marker_signals[marker_channel][start_time_ns:end_idx] = 0.4

            # Fill signals
            for start_sample, pulse in seq.pulses:
                start_time_ns = int(start_sample / self.sample_rate * 1e9)
                duration_ns = int(pulse.length / self.sample_rate * 1e9)
                end_time_ns = start_time_ns + duration_ns

                # Determine channel
                if hasattr(pulse, 'channel'):
                    channel = pulse.channel
                else:
                    channel = int(pulse.name.split("_")[-1])

                # Create pulse shape based on pulse type
                if hasattr(pulse, 'generate_samples'):
                    try:
                        pulse_envelope = _display_envelope(pulse)

                        # Skip waits / zero pulses
                        if (pulse_envelope is None or np.isscalar(pulse_envelope)
                                or len(pulse_envelope) == 0
                                or np.max(np.abs(pulse_envelope)) == 0):
                            continue

                        pulse_time = np.arange(max(0, start_time_ns),
                                               min(len(time_ns), end_time_ns))
                        if len(pulse_time) == 0:
                            continue

                        envelope_indices = np.linspace(
                            0, len(pulse_envelope) - 1, len(pulse_time), dtype=int)
                        envelope_values = pulse_envelope[envelope_indices]

                        # peak -> 0.4 above baseline (abs() so a signed envelope scales right)
                        peak = np.max(np.abs(envelope_values))
                        channel_signals[channel][pulse_time] = 0.4 * envelope_values / peak
                    except Exception as e:
                        print(
                            f"Warning3: Could not generate pulse shape for {pulse.name}, using square fallback: {e}")
                        if 0 <= start_time_ns < len(time_ns):
                            channel_signals[channel][start_time_ns:min(len(time_ns), end_time_ns)] = 0.4
                else:
                    if 0 <= start_time_ns < len(time_ns):
                        channel_signals[channel][start_time_ns:min(len(time_ns), end_time_ns)] = 0.4

            # Plot signals
            for channel in channels:
                signal_y = channel + channel_signals[channel]
                ax.plot(time_ns, signal_y, color=colors.get(channel, 'black'),
                        linewidth=2, alpha=0.8, label=channel_names.get(channel, f'Channel {channel}'))


            for mkr in markers:
                #signal_y = mkr + marker_signals[mkr]
                marker_y = marker_offset + mkr
                signal_y = marker_y + marker_signals[mkr]
                ax.plot(time_ns, signal_y, color=mk_colors.get(mkr, 'black'),
                        linewidth=2, alpha=0.8, label=marker_names.get(mkr, f'marker {mkr}'))

            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

            # Add frame counter
            ax.text(0.02, 0.98, f'Frame {frame + 1}/{len(sequences)}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(sequences),
                                       interval=interval, repeat=True, blit=False)

        return anim


class OptimizedSequence:
    """
    An optimized sequence that has been processed for hardware compatibility.
    
    This class represents a sequence that has been:
    - Validated for memory constraints
    - Split into manageable chunks
    - Optimized for general hardware constraints
    """
    
    def __init__(self, name: str, sequences: List[Sequence], metadata: Dict[str, Any] = None):
        """
        Initialize an optimized sequence.
        
        Args:
            name: Name of the sequence
            sequences: List of sequence chunks
            metadata: Additional metadata
        """
        self.name = name
        self.sequences = sequences
        self.metadata = metadata or {}
        if len(sequences) == 0:
            self.total_duration = 0.0
            self.total_samples = 0
        else:
            self.total_duration = sum(seq.duration for seq in sequences) if hasattr(sequences[0], 'duration') else 0.0
            self.total_samples = sum(len(seq.waveform) if hasattr(seq, 'waveform') else seq.length for seq in sequences)
    
    def get_chunk(self, index: int) -> Sequence:
        """
        Get a specific chunk of the sequence.
        
        Args:
            index: Chunk index
            
        Returns:
            Sequence object for the chunk
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.sequences):
            raise IndexError(f"Chunk index {index} out of range")
        return self.sequences[index]
    
    def get_chunk_count(self) -> int:
        """Get the number of chunks in this sequence."""
        return len(self.sequences)
    
    def get_total_memory_usage(self) -> int:
        """Get total memory usage in samples."""
        return self.total_samples
    
    def validate_memory_constraints(self, max_samples_per_chunk: int) -> bool:
        """
        Validate that all chunks fit within memory constraints.
        
        Args:
            max_samples_per_chunk: Maximum samples allowed per chunk
            
        Returns:
            True if all chunks are within limits
        """
        return all(len(seq.waveform) <= max_samples_per_chunk if hasattr(seq, 'waveform') else seq.length <= max_samples_per_chunk for seq in self.sequences)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of optimization results.
        
        Returns:
            Dictionary with optimization statistics
        """
        chunk_sizes = [len(seq.waveform) if hasattr(seq, 'waveform') else seq.length for seq in self.sequences]
        if len(self.sequences) == 0 or len(chunk_sizes) == 0:
            memory_efficiency = 0.0
        else:
            memory_efficiency = self.total_samples / (len(self.sequences) * max(chunk_sizes))
        return {
            "name": self.name,
            "total_chunks": len(self.sequences),
            "total_duration": self.total_duration,
            "total_samples": self.total_samples,
            "chunk_sizes": chunk_sizes,
            "memory_efficiency": memory_efficiency,
            "metadata": self.metadata
        }


class BuildError(Exception):
    """Raised when sequence building fails."""
    pass


class OptimizationError(Exception):
    """Raised when sequence optimization fails."""
    pass
