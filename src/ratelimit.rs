use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::fmt::Debug;

use opentelemetry::KeyValue;

use crate::{
    GetSamplingIntent, Threshold, SamplingIntent, ComposableParameters,
    NEVER_SAMPLE_THRESHOLD, ALWAYS_SAMPLE_THRESHOLD,
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
    current_sampling_ratio: f64,
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
            current_sampling_ratio: 1.0, // Start with full sampling ratio
        };

        RateLimitedSampler {
            delegate,
            state: Arc::new(Mutex::new(state)),
        }
    }

    /// Reset the rate limiter state.
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.last_calculation_time = Instant::now();
        state.spans_seen = 0;
        state.spans_sampled = 0;
        state.current_sampling_ratio = 1.0;
        state.incoming_rate = MovingAverage::new(10);
    }

    /// Update rate tracking and determine if we should apply rate limiting
    fn update_rate_and_check_threshold(&self, threshold: &Option<Threshold>) -> Option<Threshold> {
        let mut state = self.state.lock().unwrap();
        state.spans_seen += 1;

        // If the delegate says don't sample, respect that decision
        if threshold.is_none() || threshold == &Some(NEVER_SAMPLE_THRESHOLD) {
            return None;
        }

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
            if avg_incoming_rate > 0.0 {
                state.current_sampling_ratio = state.target_spans_per_second / avg_incoming_rate;
                // Clamp between 0.0 and 1.0
                state.current_sampling_ratio = state.current_sampling_ratio.min(1.0).max(0.0);
            }

            // Reset counters
            state.last_calculation_time = now;
            state.spans_seen = 0;
            state.spans_sampled = 0;
        }

        // Apply the sampling ratio - use probabilistic rate limiting
        let should_sample = rand::random::<f64>() < state.current_sampling_ratio;

        if should_sample {
            state.spans_sampled += 1;
            threshold.clone() // Keep original threshold from delegate
        } else {
            Some(NEVER_SAMPLE_THRESHOLD) // Force non-sampling
        }
    }
}

impl GetSamplingIntent for RateLimitedSampler {
    fn get_sampling_intent(&self, params: &ComposableParameters<'_>) -> SamplingIntent {
        // Get the delegate's sampling intent
        let delegate_intent = self.delegate.get_sampling_intent(params);

        // Apply rate limiting to the delegate's threshold
        let adjusted_threshold = self.update_rate_and_check_threshold(&delegate_intent.threshold);

        // Return a new sampling intent with possibly adjusted threshold
        // but preserving the other properties from the delegate
        SamplingIntent {
            threshold: adjusted_threshold,
            threshold_reliable: delegate_intent.threshold_reliable,
            attributes_provider: delegate_intent.attributes_provider,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ComposableSampler;
    use std::thread::sleep;

    // Helper function to create a simple test ComposableParameters
    fn test_params<'a>() -> ComposableParameters<'a> {
        ComposableParameters {
            params: &crate::Parameters {
                parent_context: None,
                trace_id: opentelemetry::trace::TraceId::INVALID,
                name: "",
                span_kind: &opentelemetry::trace::SpanKind::Internal,
                attributes: &[],
                links: &[],
            },
            parent_span_context: None,
            parent_threshold: None,
            parent_threshold_reliable: false,
        }
    }

    #[test]
    fn test_rate_limiting_with_always_on_delegate() {
        // Create a rate limiter with 10 spans per second and an AlwaysOn delegate
        let always_on = Box::new(ComposableSampler::AlwaysOn);
        let sampler = RateLimitedSampler::new(10.0, always_on);

        // Force a rate calculation by manipulating time
        {
            let mut state = sampler.state.lock().unwrap();
            state.last_calculation_time = Instant::now() - Duration::from_secs(2);
            state.spans_seen = 200; // 100 spans per second
        }

        // Generate a bunch of spans and count how many are sampled
        let mut sampled_count = 0;
        let total_spans = 100;

        for _ in 0..total_spans {
            let intent = sampler.get_sampling_intent(&test_params());

            if intent.threshold.is_some() && intent.threshold != Some(NEVER_SAMPLE_THRESHOLD) {
                sampled_count += 1;
            }
        }

        // With incoming rate of 100/s and target of 10/s, we expect about 10% sampling
        // Allow some margin for randomness
        assert!(
            sampled_count > 0 && sampled_count < 30,
            "Expected around 10% sampling rate, got {}/{}",
            sampled_count,
            total_spans
        );
    }

    #[test]
    fn test_with_always_off_delegate() {
        // Create a rate limiter with 10 spans per second and an AlwaysOff delegate
        let always_off = Box::new(ComposableSampler::AlwaysOff);
        let sampler = RateLimitedSampler::new(10.0, always_off);

        // Since the delegate says never sample, our sampler should also never sample
        for _ in 0..20 {
            let intent = sampler.get_sampling_intent(&test_params());

            // Should never sample
            assert!(intent.threshold.is_none() || intent.threshold == Some(NEVER_SAMPLE_THRESHOLD));
        }
    }

    #[test]
    fn test_reset() {
        // Create a sampler with 5 spans per second
        let always_on = Box::new(ComposableSampler::AlwaysOn);
        let sampler = RateLimitedSampler::new(5.0, always_on);

        // Force a low sampling ratio
        {
            let mut state = sampler.state.lock().unwrap();
            state.current_sampling_ratio = 0.1;
        }

        // Reset the sampler
        sampler.reset();

        // The state should be back to initial values
        let state = sampler.state.lock().unwrap();
        assert_eq!(state.current_sampling_ratio, 1.0);
        assert_eq!(state.spans_seen, 0);
        assert_eq!(state.spans_sampled, 0);
    }

    #[test]
    fn test_preserves_delegate_attributes() {
        // Create a delegate that adds attributes
        let attributes = vec![KeyValue::new("test.attribute", true)];
        let delegate = Box::new(crate::annotating_sampler(
            attributes.clone(),
            Box::new(ComposableSampler::AlwaysOn),
        ));

        let sampler = RateLimitedSampler::new(100.0, delegate); // High rate to ensure sampling

        let intent = sampler.get_sampling_intent(&test_params());

        // The attributes provider should be preserved from the delegate
        assert!(intent.attributes_provider.is_some());

        // We can't directly check the attributes here since the provider is opaque
        // In a real scenario, these would be added to the span via the ShouldSample implementation
    }

    #[test]
    fn test_rate_adaptation() {
        // Create a rate limiter with 5 spans per second
        let always_on = Box::new(ComposableSampler::AlwaysOn);
        let sampler = RateLimitedSampler::new(5.0, always_on);

        // Force a rate calculation with incoming rate of 50/s
        {
            let mut state = sampler.state.lock().unwrap();
            state.last_calculation_time = Instant::now() - Duration::from_secs(2);
            state.spans_seen = 100; // 50 spans per second
        }

        // Trigger rate recalculation
        sampler.get_sampling_intent(&test_params());

        // Verify the rate was adjusted to approximately 5/50 = 0.1
        let state = sampler.state.lock().unwrap();
        assert!(
            state.current_sampling_ratio > 0.05 && state.current_sampling_ratio < 0.15,
            "Expected sampling ratio around 0.1, got {}",
            state.current_sampling_ratio
        );
    }
}
