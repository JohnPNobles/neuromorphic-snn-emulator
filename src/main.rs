mod snn;

use embedded_hal::delay::DelayNs;
use plotters::prelude::*;
use snn::NeuromorphicCore;

struct WindowsHardwareClock;
impl DelayNs for WindowsHardwareClock {
    fn delay_ns(&mut self, _ns: u32) {}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const TOTAL_CYCLES: usize = 20;

    // Define a 3x3 Synaptic Weight Matrix
    // Rows: Input Axons (A0, A1, A2)
    // Columns: Output Neurons (N0, N1, N2)
    let synaptic_matrix: Vec<Vec<i32>> = vec![
        //  N0   N1   N2
        vec![110, 70, 10], // Axon 0 weights
        vec![30, 80, 20],  // Axon 1 weights
        vec![10, 10, 210], // Axon 2 weights (Instant trip wire for Neuron 2)
    ];

    let mut chip_core = NeuromorphicCore::new(3, 3, synaptic_matrix);
    let mut hardware_clock = WindowsHardwareClock;

    // Structured 20-cycle toy data stream demonstrating feature detection
    let incoming_spike_stream: Vec<Vec<i32>> = vec![
        // A0, A1, A2
        vec![1, 0, 0], // Cycle 01: A0 bursts
        vec![1, 1, 0], // Cycle 02: Coincidence of A0 & A1
        vec![0, 0, 0], // Cycle 03: Silence
        vec![1, 0, 0], // Cycle 04
        vec![0, 1, 0], // Cycle 05
        vec![1, 1, 0], // Cycle 06: Coincidence again
        vec![1, 0, 0], // Cycle 07
        vec![0, 0, 0], // Cycle 08
        vec![0, 1, 0], // Cycle 09
        vec![1, 1, 0], // Cycle 10
        vec![0, 0, 0], // Cycle 11
        vec![0, 0, 0], // Cycle 12
        vec![0, 0, 1], // Cycle 13: Axon 2 fires! (Anomaly test)
        vec![0, 0, 0], // Cycle 14
        vec![1, 0, 0], // Cycle 15
        vec![0, 1, 0], // Cycle 16
        vec![1, 1, 0], // Cycle 17
        vec![0, 0, 0], // Cycle 18
        vec![0, 0, 1], // Cycle 19: Axon 2 fires again!
        vec![0, 0, 0], // Cycle 20
    ];

    // Historical vectors for tracking voltage and plotting spike markers
    let mut n0_voltage = Vec::new();
    let mut n1_voltage = Vec::new();
    let mut n2_voltage = Vec::new();

    let mut n0_spikes = Vec::new();
    let mut n1_spikes = Vec::new();
    let mut n2_spikes = Vec::new();

    let time_steps: Vec<f32> = (1..=TOTAL_CYCLES).map(|x| x as f32).collect();

    println!("Executing 3x3 Parallel Neuromorphic Core Simulation...");
    println!("------------------------------------------------------------------------");

    for clock_cycle in 0..TOTAL_CYCLES {
        hardware_clock.delay_ms(1);

        let active_pins = &incoming_spike_stream[clock_cycle];
        let current_time = (clock_cycle + 1) as f32;

        // Capture internal voltage BEFORE the step calculation to see charge integration cleanly
        let v0_before = (chip_core.neurons[0].voltage as f32) / 100.0;
        let v1_before = (chip_core.neurons[1].voltage as f32) / 100.0;
        let v2_before = (chip_core.neurons[2].voltage as f32) / 100.0;

        // Step physics engine forward
        let output_spikes = chip_core.forward_clock_cycle(active_pins);

        // Record historical voltage profile
        n0_voltage.push(v0_before);
        n1_voltage.push(v1_before);
        n2_voltage.push(v2_before);

        // Record specific coordinate locations where spikes triggered
        if output_spikes[0] {
            n0_spikes.push((current_time, 2.0f32));
        }
        if output_spikes[1] {
            n1_spikes.push((current_time, 2.0f32));
        }
        if output_spikes[2] {
            n2_spikes.push((current_time, 2.0f32));
        }

        println!(
            "Cycle {:02} | Pins: {:?} | N0: {:.2}V (Spike: {:5}) | N1: {:.2}V (Spike: {:5}) | N2: {:.2}V (Spike: {:5})",
            clock_cycle + 1,
            active_pins,
            v0_before,
            output_spikes[0],
            v1_before,
            output_spikes[1],
            v2_before,
            output_spikes[2]
        );
    }

    // --- RENDER DUAL MULTI-NEURON GRAPH ---
    let root = BitMapBackend::new("neuron_voltage.png", (1024, 512)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "3x3 Neuromorphic Crossbar Silicon Emulation",
            ("sans-serif", 24).into_font(),
        )
        .margin(15)
        .x_label_area_size(45)
        .y_label_area_size(45)
        .build_cartesian_2d(1.0f32..TOTAL_CYCLES as f32, -0.2f32..2.6f32)?;

    chart
        .configure_mesh()
        .x_desc("Simulation Timestep (us)")
        .y_desc("Membrane Potential (V)")
        .draw()?;

    // 1. Static Hardware Threshold Line
    chart
        .draw_series(LineSeries::new(
            vec![(1.0, 2.0), (TOTAL_CYCLES as f32, 2.0)],
            &BLACK.mix(0.4),
        ))?
        .label("Hardware Threshold (2.0V)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK.mix(0.4)));

    // 2. Neuron 0 Plotting (Red Line + Upward Triangles for Spikes)
    chart
        .draw_series(LineSeries::new(
            time_steps.iter().cloned().zip(n0_voltage),
            &RED,
        ))?
        .label("Neuron 0 (A0 Tracker)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(
        n0_spikes
            .iter()
            .map(|&coord| TriangleMarker::new(coord, 6, &RED)),
    )?;

    // 3. Neuron 1 Plotting (Blue Line + Circles for Spikes)
    chart
        .draw_series(LineSeries::new(
            time_steps.iter().cloned().zip(n1_voltage),
            &BLUE,
        ))?
        .label("Neuron 1 (Coincidence Detector)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Mix with 0.4 opacity so shapes behind it bleed through smoothly
    chart.draw_series(
        n1_spikes
            .iter()
            .map(|&coord| Circle::new(coord, 5, BLUE.mix(0.4).filled())),
    )?;

    // 4. Neuron 2 Plotting (Green Line + Crosses for Spikes)
    chart
        .draw_series(LineSeries::new(
            time_steps.iter().cloned().zip(n2_voltage),
            &GREEN.mix(0.8),
        ))?
        .label("Neuron 2 (Axon 2 Trip Alarm)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN.mix(0.8)));

    // FIX: Pass &GREEN directly into Cross::new
    chart.draw_series(n2_spikes.iter().map(|&coord| Cross::new(coord, 5, &GREEN)))?;
    // Draw Legend Box
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.9))
        .border_style(&BLACK.mix(0.2))
        .draw()?;

    root.present()?;
    println!("\nSuccess! Architectural graph rendered to 'neuron_voltage.png'.");
    Ok(())
}
