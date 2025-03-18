use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::{
    ComposableParameters, GetSamplingIntent, SamplingIntent, Threshold, ALWAYS_SAMPLE_THRESHOLD,
};

/// Implementation of a moving average to track span rates
#[derive(Debug, Clone)]
struct MovingAverage {
    window_size: usize,
    values: Vec<f64>,
    current_index: usize,
    sum: f64,
    count: usize,
}

impl MovingAverage {
    fn new(window_size: usize) -> Self {
        MovingAverage {
            window_size,
            values: vec![0.0; window_size],
            current_index: 0,
            sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, value: f64) {
        if self.count < self.window_size {
            self.count += 1;
        } else {
            self.sum -= self.values[self.current_index];
        }

        self.values[self.current_index] = value;
        self.sum += value;
        self.current_index = (self.current_index + 1) % self.window_size;
    }

    fn average(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }
}

/// State maintained by the rate limiter
#[derive(Debug)]
struct RateLimiterState {
    // Target maximum sampling rate (spans per second)
    target_spans_per_second: f64,
    // Last time the rate was calculated
    last_calculation_time: Instant,
    // Number of spans seen since last calculation
    spans_seen: u64,
    // Number of spans sampled since last calculation
    spans_sampled: u64,
    // Moving average of incoming spans per second
    incoming_rate: MovingAverage,
    // Current sampling ratio to apply (between 0 and 1)
    current_sampling_ratio: Threshold,
}

/// RateLimitedSampler implements a composite sampler that helps control the average rate
/// of sampled spans while allowing another sampler (the delegate) to provide sampling hints.
///
/// This implements the ConsistentRateLimiting strategy described in OTEP-0250.
#[derive(Debug, Clone)]
pub struct RateLimitedSampler {
    // The delegate sampler that provides initial sampling decisions
    delegate: Box<dyn GetSamplingIntent>,
    // State for rate limiting, shared across clones
    state: Arc<Mutex<RateLimiterState>>,
}

impl RateLimitedSampler {
    /// Create a new RateLimitedSampler with the specified rate limit.
    ///
    /// # Arguments
    ///
    /// * `target_spans_per_second` - The maximum number of spans to sample per second
    /// * `delegate` - The delegate sampler that provides initial sampling decisions
    pub fn new(target_spans_per_second: f64, delegate: Box<dyn GetSamplingIntent>) -> Self {
        let state = RateLimiterState {
            target_spans_per_second,
            last_calculation_time: Instant::now(),
            spans_seen: 0,
            spans_sampled: 0,
            incoming_rate: MovingAverage::new(10), // 10-sample window for rate averaging
            current_sampling_ratio: ALWAYS_SAMPLE_THRESHOLD, // Start with full sampling ratio
        };

        RateLimitedSampler {
            delegate,
            state: Arc::new(Mutex::new(state)),
        }
    }

    /// Update rate tracking and determine if we should apply rate limiting
    fn update_rate_and_check_threshold(
        &self,
        delegate_threshold: Option<&Threshold>,
    ) -> Option<Threshold> {
        delegate_threshold.cloned().map(|dth| {
            let mut state = self.state.lock().unwrap();
            state.spans_seen += 1;

            // Update rate calculations periodically
            let now = Instant::now();
            let elapsed = now.duration_since(state.last_calculation_time);

            if elapsed >= Duration::from_secs(1) {
                // Calculate incoming spans per second
                let spans_per_sec = state.spans_seen as f64 / elapsed.as_secs_f64();

                // Update moving average of incoming rate
                state.incoming_rate.add(spans_per_sec);

                // Calculate average incoming rate
                let avg_incoming_rate = state.incoming_rate.average();

                // Adjust sampling ratio to meet target rate
                let new_ratio = if avg_incoming_rate > 0.0 {
                    let avg = state.target_spans_per_second / avg_incoming_rate;
                    // Clamp between 0.0 and 1.0
                    avg.min(1.0)
                } else {
                    1.0
                };

                // Reset counters
                state.last_calculation_time = now;
                state.spans_seen = 0;
                state.spans_sampled = 0;
                state.current_sampling_ratio = Threshold::from(new_ratio);
            }

            // If the delegate's threshold is more restrictive than
            // the current limit, use it, increase the threshold.
            if dth.0 > state.current_sampling_ratio.0 {
                dth
            } else {
                state.current_sampling_ratio.clone()
            }
        })
    }
}

impl GetSamplingIntent for RateLimitedSampler {
    fn get_sampling_intent(&self, params: &ComposableParameters<'_>) -> SamplingIntent {
        // Get the delegate's sampling intent
        let delegate_intent = self.delegate.get_sampling_intent(params);

        // Apply rate limiting to the delegate's threshold
        let adjusted_threshold =
            self.update_rate_and_check_threshold(delegate_intent.threshold.as_ref());

        // Return a new sampling intent with possibly adjusted threshold
        // but preserving the other properties from the delegate
        SamplingIntent {
            threshold: adjusted_threshold,
            threshold_reliable: delegate_intent.threshold_reliable,
            attributes_provider: delegate_intent.attributes_provider,
        }
    }
}
