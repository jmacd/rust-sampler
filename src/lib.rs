use opentelemetry::{
    trace::{
        Link,
	SamplingDecision,
	SamplingResult,
	SpanKind,
	TraceId,
	TraceState,
    },
    Context, KeyValue,
};

// OTEP 235 constants

/// DefaultSamplingPrecision is the number of hexadecimal
/// digits of precision used to expressed the samplling probability.
const DEFAULT_SAMPLING_PRECISION: u32 = 4;

/// MinSupportedProbability is the smallest probability that
/// can be encoded by this implementation, and it defines the
/// smallest interval between probabilities across the range.
/// The largest supported probability is (1-MinSupportedProbability).
///
/// This value corresponds with the size of a float64
/// significand, because it simplifies this implementation to
/// restrict the probability to use 52 bits (vs 56 bits).
const MIN_SUPPORTED_PROBABILITY: f64 = 1f64 / (MAX_ADJUSTED_COUNT as f64);

// maxSupportedProbability is the number closest to 1.0 (i.e.,
// near 99.999999%) that is not equal to 1.0 in terms of the
// float64 representation, having 52 bits of significand.
// Other ways to express this number:
//
//   0x1.ffffffffffffe0p-01
//   0x0.fffffffffffff0p+00
//   math.Nextafter(1.0, 0.0)
const MAX_SUPPORTED_PROBABILITY: f64 = 1f64 - (1f64 / ((1u64<<52) as f64));

// maxAdjustedCount is the inverse of the smallest
// representable sampling probability, it is the number of
// distinct 56 bit values.
const MAX_ADJUSTED_COUNT: u64 = 1u64 << 56;

// randomnessMask is a mask that selects the least-significant
// 56 bits of a uint64.
//const RANDOMNESS_MASK: u64 = MAX_ADJUSTED_COUNT - 1;

// NEVER_SAMPLE_THRESHOLD indicates a span that should not be sampled.
// This is equivalent to sampling with 0% probability.
const NEVER_SAMPLE_THRESHOLD: Threshold = Threshold(1u64<<56);

// ALWAYS_SAMPLE_THREHSOLD indicates to sample with 100% probability.
const ALWAYS_SAMPLE_THRESHOLD: Threshold = Threshold(0);

/// Threshold is computed from TraceState fields and/or f64 values.
/// As in OTEP 235, the 0 value means rejecting 0 spans.
/// The value must be <= NEVER_SAMPLE_THRESHOLD.
#[derive(Clone)]
pub struct Threshold(u64);

/// A ComposableSampler implements the built-in composable samplers.
/// GetSamplingIntent is implemented for these according to OTEP 4321.
#[derive(Debug, Clone)]
pub enum ComposableSampler {
    AlwaysOn,
    AlwaysOff,
    //Annotating,
    ParentThreshold,
    RuleBased,
    TraceIdRatio(Threshold),
}

/// SamplingIntent is an individual part in a composable decision.
pub struct SamplingIntent {
	threshold:         Threshold,          // i.e., sampling probability, implies record & export when...
	threshold_reliable: bool,           // whether the threshold is reliable

	// TODO
	//Attributes        AttributesFunc // add attributes the span
	//TraceState        TraceStateFunc // update the tracestate
}

/// A CompositeSampler implements the original ShouldSample interface
/// used by the OTel SDK.  This refines the basic sampler with
/// composable samplers, which are more efficient when multiple
/// logical expressions are used to evalute the sampler.
#[derive(Debug, Clone)]
pub struct CompositeSampler {
    sampler: Box<dyn GetSamplingIntent>,
}

/// GetSamplingIntent is part in a composable sampler decision.
pub trait GetSamplingIntent: CloneGetSamplingIntent + Send + Sync + std::fmt::Debug {
    fn get_sampling_intent(
        &self,
	_parent_context: Option<&Context>,
        _trace_id: TraceId,
        _name: &str,
        _span_kind: &SpanKind,
        _attributes: &[KeyValue],
        _links: &[Link],
    ) -> SamplingIntent;
}

/// This trait should not be used directly instead users should use [`GetSamplingIntent`].
pub trait CloneGetSamplingIntent {
    fn box_clone(&self) -> Box<dyn GetSamplingIntent>;
}

impl<T> CloneGetSamplingIntent for T
where
    T: GetSamplingIntent + Clone + 'static,
{
    fn box_clone(&self) -> Box<dyn GetSamplingIntent> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn GetSamplingIntent> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

// CompositeSampler

impl CompositeSampler {
    pub fn new(sampler: Box<dyn GetSamplingIntent>) -> Self {
	CompositeSampler{
	    sampler,
	}
    }
}

impl opentelemetry_sdk::trace::ShouldSample for CompositeSampler {
    fn should_sample(
        &self,
	_parent_context: Option<&Context>,
        _trace_id: TraceId,
        _name: &str,
        _span_kind: &SpanKind,
        _attributes: &[KeyValue],
        _links: &[Link],
    ) -> SamplingResult {

	SamplingResult {
	    decision: SamplingDecision::RecordAndSample,
	    attributes: vec![],
	    trace_state: TraceState::default(),
	}
    }
}

// ComposableSampler

impl GetSamplingIntent for ComposableSampler {
    fn get_sampling_intent(
        &self,
	_parent_context: Option<&Context>,
        _trace_id: TraceId,
        _name: &str,
        _span_kind: &SpanKind,
        _attributes: &[KeyValue],
        _links: &[Link],
    ) -> SamplingIntent {
        match self {
            // Always sample the trace
            ComposableSampler::AlwaysOn => SamplingIntent {
		threshold: ALWAYS_SAMPLE_THRESHOLD,
		threshold_reliable: true,
	    },
            ComposableSampler::AlwaysOff => SamplingIntent {
		threshold: NEVER_SAMPLE_THRESHOLD,
		threshold_reliable: false,
	    },
            ComposableSampler::TraceIdRatio(threshold) => SamplingIntent {
		threshold: threshold.clone(),
		threshold_reliable: true,
	    },
            // ComposableSampler::Annotating => SamplingIntent {
	    // 	threshold: NEVER_SAMPLE_THRESHOLD,
	    // 	threshold_reliable: false,
	    // },
            ComposableSampler::ParentThreshold => SamplingIntent {
		threshold: NEVER_SAMPLE_THRESHOLD,
		threshold_reliable: false,
	    },
            ComposableSampler::RuleBased => SamplingIntent {
		threshold: NEVER_SAMPLE_THRESHOLD,
		threshold_reliable: false,
	    },
	}
    }    
}

// Threshold
impl Threshold {
    pub fn from(fraction: f64) -> Self {
	Self::from_with_precision(fraction, DEFAULT_SAMPLING_PRECISION)
    }

    pub fn from_with_precision(fraction: f64, precision: u32) -> Self {
	const MAXP: u32 = 14; // maximum precision is 56 bits

	if fraction > MAX_SUPPORTED_PROBABILITY {
	    return ALWAYS_SAMPLE_THRESHOLD;
	}

	if fraction < MIN_SUPPORTED_PROBABILITY {
	    return NEVER_SAMPLE_THRESHOLD;
	}

	// Calculate the amount of precision needed to encode the
	// threshold with reasonable precision.  The expression
	// leading_zeros calculates log2() and divides by -4 for
	// (the number of bits per hex digit) for the number of
	// leading digits that will be `f`.
	//
	// We know that `exp <= 0`.  If `exp <= -4`, there will be a
	// leading hex `f`.  For every multiple of -4, another leading
	// `f` appears, so this raises precision accordingly.
	let leading_fs = (fraction.log2() / -4.0).floor() as u32;
	let final_precision = std::cmp::min(MAXP, std::cmp::max(1u32, precision+leading_fs));

	// // Compute the threshold
	let scaled = (fraction * MAX_ADJUSTED_COUNT as f64).round() as u64;
	let mut threshold = MAX_ADJUSTED_COUNT - scaled;

	// Round to the specified precision, if less than the maximum.
	// Here, 4 is the number of bits per hex digit.
	let shift = 4 * (MAXP - final_precision);
	if shift != 0 {
	    let half = 1u64 << (shift - 1);
	    threshold += half;
	    threshold >>= shift;
	    threshold <<= shift;
	}
	Threshold(threshold)
    }

    pub fn to_string(&self) -> String {
	format!("{:?}", self)
    }
}

impl std::fmt::Debug for Threshold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
	if self.0 == 0 {
	    // 0 is a recognized special case.
	    f.write_str("0")
	} else if self.0 >= MAX_ADJUSTED_COUNT {
	    f.write_str("never_sampled")
	} else {
	    // this branch removes trailing zeros.
	    let x = MAX_ADJUSTED_COUNT + self.0;
	    let s = format!("{:x}", x);
	    let mut t = s.as_bytes();
	    t = &t[1..];
	    loop {
		if t[t.len()-1] == ('0' as u8) {
		    t = &t[0..t.len()-1];
		} else {
		    break;
		}
	    }
	    let s = std::str::from_utf8(t).unwrap();
	    f.write_str(s)
	}
    }    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_to_string() {
	assert_eq!(ALWAYS_SAMPLE_THRESHOLD.to_string(), "0");
	assert_eq!(Threshold::from(2.0).to_string(), "0");
	assert_eq!(Threshold::from(1.0).to_string(), "0");
	assert_eq!(Threshold::from(0.5).to_string(), "8");
	assert_eq!(Threshold::from(0.25).to_string(), "c");
	assert_eq!(Threshold::from(0.01).to_string(), "fd70a");
	assert_eq!(Threshold::from(MIN_SUPPORTED_PROBABILITY).to_string(), "ffffffffffffff");

	assert_eq!(NEVER_SAMPLE_THRESHOLD.to_string(), "never_sampled");
	assert_eq!(Threshold(MAX_ADJUSTED_COUNT).to_string(), "never_sampled");
	assert_eq!(Threshold(u64::MAX).to_string(), "never_sampled");

	assert_eq!(Threshold::from_with_precision(1f64/3f64, 14).to_string(), "aaaaaaaaaaaaac");
	assert_eq!(Threshold::from_with_precision(1f64/3f64, 10).to_string(), "aaaaaaaaab");
	assert_eq!(Threshold::from_with_precision(1f64/3f64, 2).to_string(), "ab");
	assert_eq!(Threshold::from_with_precision(1f64/3f64, 1).to_string(), "b");

	assert_eq!(Threshold::from_with_precision(0.01, 8).to_string(), "fd70a3d71");
	assert_eq!(Threshold::from_with_precision(0.99, 8).to_string(), "028f5c29");
    }
}
