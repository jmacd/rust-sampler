// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// Nit: why doesn't OTel-Rust use copyright headers?
use opentelemetry::{
    trace::{
        Link, SamplingDecision, SamplingResult, SpanContext, SpanKind, TraceContextExt, TraceId,
        TraceState,
    },
    Context, KeyValue,
};

// OTEP 235 constants

/// DefaultSamplingPrecision is the number of hexadecimal
/// digits of precision used to expressed the samplling probability.
const DEFAULT_SAMPLING_PRECISION: u32 = 4;

/// OTEP 235 limits precision to 56 bits.
const MAX_SAMPLING_PRECISION: u32 = 14;

/// This is the smallest probability that can be encoded by this
/// implementation, and it defines the smallest interval between
/// probabilities across the range.  The largest supported probability
/// is (1-MinSupportedProbability).
///
/// This value corresponds with the size of a float64
/// significand, because it simplifies this implementation to
/// restrict the probability to use 52 bits (vs 56 bits).
const MIN_SUPPORTED_PROBABILITY: f64 = 1f64 / (MAX_ADJUSTED_COUNT as f64);

/// This is the number closest to 1.0 (i.e., near 99.999999%) that is
/// not equal to 1.0 in terms of the float64 representation, having 52
/// bits of significand.  Other ways to express this number:
///
///   0x1.ffffffffffffe0p-01
///   0x0.fffffffffffff0p+00
///   math.Nextafter(1.0, 0.0)
const MAX_SUPPORTED_PROBABILITY: f64 = 1f64 - (1f64 / ((1u64 << 52) as f64));

/// This is the inverse of the smallest representable sampling
/// probability, it is the number of distinct 56 bit values.
const MAX_ADJUSTED_COUNT: u64 = 1u64 << 56;

/// This indicates a span that should not be sampled.  This is
/// equivalent to sampling with 0% probability.
const NEVER_SAMPLE_THRESHOLD: Threshold = Threshold(1u64 << 56);

/// This indicates to sample with 100% probability.
const ALWAYS_SAMPLE_THRESHOLD: Threshold = Threshold(0);

/// Threshold is computed from TraceState fields and/or f64 values.
/// As in OTEP 235, the 0 value means rejecting 0 spans.
/// The value must be <= NEVER_SAMPLE_THRESHOLD.
#[derive(Clone, PartialEq)]
pub struct Threshold(u64);

/// Randomness is computed from TraceState and/or TraceID values.
#[derive(Clone, PartialEq)]
pub struct Randomness(u64);

/// AttributesProvider is a trait for providing attributes in a sampling decision
pub trait AttributesProvider: AttributesProviderClone + Send + Sync + std::fmt::Debug {
    fn get_attributes(&self) -> Vec<KeyValue>;
}

/// This trait supports cloning Box<dyn AttributesProvider>
pub trait AttributesProviderClone {
    fn box_clone(&self) -> Box<dyn AttributesProvider>;
}

impl<T> AttributesProviderClone for T
where
    T: AttributesProvider + Clone + 'static,
{
    fn box_clone(&self) -> Box<dyn AttributesProvider> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AttributesProvider> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// A simple implementation of AttributesProvider that returns a fixed list of attributes
#[derive(Debug, Clone)]
pub struct StaticAttributesProvider {
    attributes: Vec<KeyValue>,
}

impl AttributesProvider for StaticAttributesProvider {
    fn get_attributes(&self) -> Vec<KeyValue> {
        self.attributes.clone()
    }
}

/// A ComposableSampler implements the built-in composable samplers.
/// GetSamplingIntent is implemented for these according to OTEP 4321.
#[derive(Debug, Clone)]
pub enum ComposableSampler {
    AlwaysOn,
    AlwaysOff,
    TraceIdRatio(Threshold),
    ParentThreshold,
    RuleBased(Vec<RuleAndPredicate>),
    Annotating(Box<dyn AttributesProvider>, Box<dyn GetSamplingIntent>),
}

/// SamplingIntent is an individual part in a composable decision.
pub struct SamplingIntent {
    threshold: Option<Threshold>,
    attributes_provider: Option<Box<dyn AttributesProvider>>,
}

/// Parameters are the arguments to GetSamplingIntent.
#[allow(unused)]
pub struct Parameters<'a> {
    parent_context: Option<&'a Context>,
    trace_id: TraceId,
    name: &'a str,
    span_kind: &'a SpanKind,
    attributes: &'a [KeyValue],
    links: &'a [Link],
}

#[allow(unused)]
pub struct ComposableParameters<'a> {
    params: &'a Parameters<'a>,
    parent_span_context: Option<&'a SpanContext>,
    parent_threshold: Option<Threshold>,
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
    fn get_sampling_intent(&self, params: &ComposableParameters<'_>) -> SamplingIntent;
}

/// This trait supports cloning Box<dyn GetSamplingIntent>.
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
        CompositeSampler { sampler }
    }
}

impl opentelemetry_sdk::trace::ShouldSample for CompositeSampler {
    fn should_sample(
        &self,
        parent_context: Option<&Context>,
        trace_id: TraceId,
        name: &str,
        span_kind: &SpanKind,
        attributes: &[KeyValue],
        links: &[Link],
    ) -> SamplingResult {
        let params = Parameters {
            parent_context,
            trace_id,
            name,
            span_kind,
            attributes,
            links,
        };

        parent_context
            .filter(|cx| cx.has_active_span())
            .map_or_else(
                || self.should_sample_parent(&params, None),
                |ctx| {
                    let span = ctx.span();
                    let parent_span_context = span.span_context();

                    self.should_sample_parent(&params, Some(parent_span_context))
                },
            )
    }
}

impl CompositeSampler {
    fn should_sample_parent(
        &self,
        params: &Parameters<'_>,
        parent_span_context: Option<&SpanContext>,
    ) -> SamplingResult {
        parent_span_context.map_or_else(
            || self.should_sample_otts(params, parent_span_context, None),
            |sctx| {
                self.should_sample_otts(params, parent_span_context, sctx.trace_state().get("ot"))
            },
        )
    }

    fn should_sample_otts(
        &self,
        params: &Parameters<'_>,
        parent_span_context: Option<&SpanContext>,
        otts_str: Option<&str>,
    ) -> SamplingResult {
        // Note: I added TraceState::from_str_delimited() for this
        // purpose: would be more proper to add a new types
        // representing the two level of trace state.
        let otts = otts_str
            .map(|s| TraceState::from_str_delimited(s, ':', ';').ok())
            .flatten();

        let rv_value = otts.as_ref().map(|ts| ts.get("rv")).flatten();
        let th_value = otts.as_ref().map(|ts| ts.get("th")).flatten();

        let randomness = rv_value
            .map(|rv| Randomness::from_rv_value(rv))
            .flatten()
            .or_else(|| Some(Randomness::from_trace_id(params.trace_id)))
            .unwrap();

        let threshold = th_value.map(|th| Threshold::from_th_value(th)).flatten();

        self.should_sample_randomness_threshold(params, parent_span_context, randomness, threshold)
    }

    fn should_sample_randomness_threshold(
        &self,
        params: &Parameters<'_>,
        parent_span_context: Option<&SpanContext>,
        randomness: Randomness,
        mut parent_threshold: Option<Threshold>,
    ) -> SamplingResult {
        let mut update_trace_flag = false;
        let mut erase_threshold = false;
        let mut threshold_reliable = false;

        let flag_sampled = parent_span_context
            .map(|psc| psc.is_sampled())
            .or_else(|| Some(false))
            .unwrap();

        if let Some(parent_threshold) = parent_threshold.as_ref() {
            let threshold_sampled = parent_threshold.0 <= randomness.0;

            match (threshold_sampled, flag_sampled) {
                (true, true) => {
                    // The two agree to sample.  Threshold is reliable.
                    threshold_reliable = true;
                }
                (true, false) => {
                    // Threshold says sampled, flag says not.
                    update_trace_flag = true;
                }
                (false, true) => {
                    // Flag says sampled, threshold says not.  The threshold should not propagate.
                    erase_threshold = true;
                }
                (false, false) => {
                    // The two agree not to sample.  The threshold should not propagate.
                    erase_threshold = true;
                }
            }
        }
        if !threshold_reliable {
            parent_threshold = None;
        }

        let cparams = ComposableParameters {
            params,
            parent_span_context,
            parent_threshold,
        };
        let intent = self.sampler.get_sampling_intent(&cparams);

        let sampled = intent
            .threshold
            .map(|th| th.0 <= randomness.0)
            .or(Some(false))
            .unwrap();

        // TODO: recombine tracestate; evaluate attributes funcs.
        (_, _) = (update_trace_flag, erase_threshold);

        let decision = if sampled {
            SamplingDecision::RecordAndSample
        } else {
            SamplingDecision::Drop
        };

        // Only collect attributes if we're sampling this span

        let attributes = sampled.then().then();

	//     if sampled && intent.attributes_provider.is_some() {
        //     intent.attributes_provider.get_attributes()
        // } else {
        //     vec![]
        // };

        SamplingResult {
            decision: decision,
            attributes,
            trace_state: TraceState::default(),
        }
    }
}

// A combined attributes provider that merges attributes from multiple providers
#[derive(Debug, Clone)]
struct CombinedAttributesProvider {
    a: Box<dyn AttributesProvider>,
    b: Box<dyn AttributesProvider>,
}

impl AttributesProvider for CombinedAttributesProvider {
    fn get_attributes(&self) -> Vec<KeyValue> {
        let mut result = Vec::new();
        result.extend(self.a.get_attributes());
        result.extend(self.b.get_attributes());
        result
    }
}

// ComposableSampler

impl GetSamplingIntent for ComposableSampler {
    fn get_sampling_intent(&self, params: &ComposableParameters<'_>) -> SamplingIntent {
        match self {
            ComposableSampler::AlwaysOn => SamplingIntent {
                threshold: Some(ALWAYS_SAMPLE_THRESHOLD),
                attributes_provider: None,
            },
            ComposableSampler::AlwaysOff => SamplingIntent {
                threshold: None,
                attributes_provider: None,
            },
            ComposableSampler::TraceIdRatio(threshold) => SamplingIntent {
                threshold: Some(threshold.clone()),
                attributes_provider: None,
            },
            ComposableSampler::ParentThreshold => SamplingIntent {
                threshold: params.parent_threshold.clone(),
                attributes_provider: None,
            },
            ComposableSampler::RuleBased(rules) => {
                // Evaluate rules in order
                for rule in rules {
                    if rule.predicate.decide(params) {
                        return rule.sampler.get_sampling_intent(params);
                    }
                }
                // Default case when no rules match
                SamplingIntent {
                    threshold: None,
                    attributes_provider: None,
                }
            }
            ComposableSampler::Annotating(provider, delegate) => {
                let delegate_intent = delegate.get_sampling_intent(params);

                // Create a combined attributes provider if delegate has attributes,
                // otherwise just use our provider
                let attributes_provider =
                    if let Some(delegate_provider) = delegate_intent.attributes_provider {
                        Some(Box::new(CombinedAttributesProvider {
                            a: delegate_provider,
                            b: provider.clone(),
                        }) as Box<dyn AttributesProvider>)
                    } else {
                        Some(provider.clone())
                    };

                // Use the delegate's threshold
                SamplingIntent {
                    threshold: delegate_intent.threshold,
                    attributes_provider,
                }
            }
        }
    }
}

// Threshold

impl Threshold {
    pub fn from(fraction: f64) -> Self {
        Self::from_with_precision(fraction, DEFAULT_SAMPLING_PRECISION)
    }

    pub fn from_with_precision(fraction: f64, precision: u32) -> Self {
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
        let final_precision = std::cmp::min(
            MAX_SAMPLING_PRECISION,
            std::cmp::max(1u32, precision + leading_fs),
        );

        // // Compute the threshold
        let scaled = (fraction * MAX_ADJUSTED_COUNT as f64).round() as u64;
        let mut threshold = MAX_ADJUSTED_COUNT - scaled;

        // Round to the specified precision, if less than the maximum.
        // Here, 4 is the number of bits per hex digit.
        let shift = 4 * (MAX_SAMPLING_PRECISION - final_precision);
        if shift != 0 {
            let half = 1u64 << (shift - 1);
            threshold += half;
            threshold >>= shift;
            threshold <<= shift;
        }
        Self(threshold)
    }

    /// Extracts a threshold from an OTel tracestate 'th' value.
    pub fn from_th_value(th: &str) -> Option<Self> {
        if th.len() == 0 || th.len() > MAX_SAMPLING_PRECISION as usize {
            None
        } else {
            u64::from_str_radix(th, 16).ok().map(|x| {
                let shift = (MAX_SAMPLING_PRECISION as usize - th.len()) * 4;
                Threshold(x << shift)
            })
        }
    }

    pub fn to_string(&self) -> String {
        format!("{:?}", self)
    }
}

// Randomness

impl Randomness {
    /// Extracts the least-significant 56 bits of the TraceId.
    pub fn from_trace_id(tid: TraceId) -> Randomness {
        // Note there's no way to access the u128 underneath, otherwise
        // this could be an arithmetic expression,
        //
        //  u64::from(tid.to_u128()) & (MAX_ADJUSTED_COUNT - 1)
        //
        // i.e., (trace_id & 0xffffffffffffff) as u64.
        let by16 = tid.to_bytes();
        Randomness(u64::from_be_bytes([
            0, by16[9], by16[10], by16[11], by16[12], by16[13], by16[14], by16[15],
        ]))
    }

    /// Extracts explicit randomness, expecting exactly 14 hex digits.
    pub fn from_rv_value(rv: &str) -> Option<Randomness> {
        // Use explicit randomness.
        if rv.len() != MAX_SAMPLING_PRECISION as usize {
            None
        } else {
            u64::from_str_radix(rv, 16).ok().map(|x| Randomness(x))
        }
    }
}

// fmt::Debug

impl std::fmt::Debug for Randomness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let x = MAX_ADJUSTED_COUNT + self.0;
        let xtra = format!("{:x}", x);
        f.write_str(xtra.as_str().strip_prefix("1").unwrap())
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
                if t[t.len() - 1] == ('0' as u8) {
                    t = &t[0..t.len() - 1];
                } else {
                    break;
                }
            }
            let s = std::str::from_utf8(t).unwrap();
            f.write_str(s)
        }
    }
}

// Predicate trait for rule-based sampling decisions
pub trait Predicate: PredicateClone + Send + Sync + std::fmt::Debug {
    fn decide(&self, params: &ComposableParameters<'_>) -> bool;
    fn description(&self) -> String;
}

// Helper trait for cloning boxed Predicates
pub trait PredicateClone {
    fn box_clone(&self) -> Box<dyn Predicate>;
}

impl<T> PredicateClone for T
where
    T: Predicate + Clone + 'static,
{
    fn box_clone(&self) -> Box<dyn Predicate> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Predicate> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

// A simple predicate that always returns true
#[derive(Debug, Clone)]
pub struct TruePredicate {}

impl Predicate for TruePredicate {
    fn decide(&self, _params: &ComposableParameters<'_>) -> bool {
        true
    }

    fn description(&self) -> String {
        "true".to_string()
    }
}

// A predicate that checks if the span is a root span
#[derive(Debug, Clone)]
pub struct IsRootPredicate {}

impl Predicate for IsRootPredicate {
    fn decide(&self, params: &ComposableParameters<'_>) -> bool {
        params.parent_span_context.is_none()
    }

    fn description(&self) -> String {
        "is_root".to_string()
    }
}

// Options for configuring RuleBased sampler
pub struct RuleBasedOption(Box<dyn Fn(&mut RuleBasedConfig)>);

// A rule paired with a predicate
#[derive(Clone, Debug)]
pub struct RuleAndPredicate {
    predicate: Box<dyn Predicate>,
    sampler: Box<dyn GetSamplingIntent>,
}

// Configuration for RuleBased sampler
#[derive(Default, Clone)]
pub struct RuleBasedConfig {
    rules: Vec<RuleAndPredicate>,
    default_rule: Option<Box<dyn GetSamplingIntent>>,
}

// Helper functions to create predicates
pub fn true_predicate() -> Box<dyn Predicate> {
    Box::new(TruePredicate {})
}

pub fn is_root_predicate() -> Box<dyn Predicate> {
    Box::new(IsRootPredicate {})
}

// Configuration options for RuleBased sampler
pub fn with_rule(
    predicate: Box<dyn Predicate>,
    sampler: Box<dyn GetSamplingIntent>,
) -> RuleBasedOption {
    RuleBasedOption(Box::new(move |config: &mut RuleBasedConfig| {
        config.rules.push(RuleAndPredicate {
            predicate: predicate.clone(),
            sampler: sampler.clone(),
        });
    }))
}

pub fn with_default_rule(sampler: Box<dyn GetSamplingIntent>) -> RuleBasedOption {
    RuleBasedOption(Box::new(move |config: &mut RuleBasedConfig| {
        config.default_rule = Some(sampler.clone());
    }))
}

// Create a new RuleBased sampler
pub fn rule_based(options: Vec<RuleBasedOption>) -> ComposableSampler {
    let mut config = RuleBasedConfig::default();

    for option in options {
        (option.0)(&mut config);
    }

    if let Some(default_rule) = config.default_rule {
        config.rules.push(RuleAndPredicate {
            predicate: true_predicate(),
            sampler: default_rule,
        });
    }

    ComposableSampler::RuleBased(config.rules)
}

// Create a parent-based composable sampler
pub fn composable_parent_based(root: Box<dyn GetSamplingIntent>) -> ComposableSampler {
    rule_based(vec![
        with_rule(is_root_predicate(), root),
        with_default_rule(Box::new(ComposableSampler::ParentThreshold)),
    ])
}

// A predicate that negates another predicate
#[derive(Debug, Clone)]
pub struct NegatedPredicate {
    original: Box<dyn Predicate>,
}

impl Predicate for NegatedPredicate {
    fn decide(&self, params: &ComposableParameters<'_>) -> bool {
        !self.original.decide(params)
    }

    fn description(&self) -> String {
        format!("not({})", self.original.description())
    }
}

// A predicate that checks if span name matches
#[derive(Debug, Clone)]
pub struct SpanNamePredicate {
    name: String,
}

impl Predicate for SpanNamePredicate {
    fn decide(&self, params: &ComposableParameters<'_>) -> bool {
        self.name == params.params.name
    }

    fn description(&self) -> String {
        format!("Span.Name=={}", self.name)
    }
}

// A predicate that checks if span kind matches
#[derive(Debug, Clone)]
pub struct SpanKindPredicate {
    kind: SpanKind,
}

impl Predicate for SpanKindPredicate {
    fn decide(&self, params: &ComposableParameters<'_>) -> bool {
        self.kind == *params.params.span_kind
    }

    fn description(&self) -> String {
        format!("Span.Kind=={:?}", self.kind)
    }
}

// A predicate that checks if the parent context is remote
#[derive(Debug, Clone)]
pub struct IsRemotePredicate {}

impl Predicate for IsRemotePredicate {
    fn decide(&self, params: &ComposableParameters<'_>) -> bool {
        params
            .parent_span_context
            .map(|ctx| ctx.is_remote())
            .unwrap_or(false)
    }

    fn description(&self) -> String {
        "remote?".to_string()
    }
}

// A predicate that checks if the parent context is local
#[derive(Debug, Clone)]
pub struct IsLocalPredicate {}

impl Predicate for IsLocalPredicate {
    fn decide(&self, params: &ComposableParameters<'_>) -> bool {
        params
            .parent_span_context
            .map(|ctx| ctx.is_valid() && !ctx.is_remote())
            .unwrap_or(false)
    }

    fn description(&self) -> String {
        "local?".to_string()
    }
}

// Helper functions to create predicates
pub fn negate_predicate(original: Box<dyn Predicate>) -> Box<dyn Predicate> {
    Box::new(NegatedPredicate { original })
}

pub fn span_name_predicate(name: String) -> Box<dyn Predicate> {
    Box::new(SpanNamePredicate { name })
}

pub fn span_kind_predicate(kind: SpanKind) -> Box<dyn Predicate> {
    Box::new(SpanKindPredicate { kind })
}

pub fn is_remote_predicate() -> Box<dyn Predicate> {
    Box::new(IsRemotePredicate {})
}

pub fn is_local_predicate() -> Box<dyn Predicate> {
    Box::new(IsLocalPredicate {})
}

// Helper function to create an annotating sampler
pub fn annotating_sampler(
    attributes: Vec<KeyValue>,
    delegate: Box<dyn GetSamplingIntent>,
) -> ComposableSampler {
    ComposableSampler::Annotating(Box::new(StaticAttributesProvider { attributes }), delegate)
}

// Helper function to create a simple always-on annotating sampler
pub fn always_on_with_attributes(attributes: Vec<KeyValue>) -> ComposableSampler {
    annotating_sampler(attributes, Box::new(ComposableSampler::AlwaysOn))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: this package makes testing threshold encodings natural.
    // because both use hexadecimal encoding.
    use hexf::hexf64;

    #[test]
    fn threshold_to_string() {
        assert_eq!(ALWAYS_SAMPLE_THRESHOLD.to_string(), "0");
        assert_eq!(Threshold::from(2.0).to_string(), "0");
        assert_eq!(Threshold::from(1.0).to_string(), "0");
        assert_eq!(Threshold::from(0.5).to_string(), "8");
        assert_eq!(Threshold::from(0.25).to_string(), "c");
        assert_eq!(Threshold::from(0.01).to_string(), "fd70a");
        assert_eq!(
            Threshold::from(MIN_SUPPORTED_PROBABILITY).to_string(),
            "ffffffffffffff"
        );

        assert_eq!(NEVER_SAMPLE_THRESHOLD.to_string(), "never_sampled");
        assert_eq!(Threshold(MAX_ADJUSTED_COUNT).to_string(), "never_sampled");
        assert_eq!(Threshold(u64::MAX).to_string(), "never_sampled");

        assert_eq!(
            Threshold::from_with_precision(1f64 / 3f64, 14).to_string(),
            "aaaaaaaaaaaaac"
        );
        assert_eq!(
            Threshold::from_with_precision(1f64 / 3f64, 10).to_string(),
            "aaaaaaaaab"
        );
        assert_eq!(
            Threshold::from_with_precision(1f64 / 3f64, 2).to_string(),
            "ab"
        );
        assert_eq!(
            Threshold::from_with_precision(1f64 / 3f64, 1).to_string(),
            "b"
        );

        assert_eq!(
            Threshold::from_with_precision(0.01, 8).to_string(),
            "fd70a3d71"
        );
        assert_eq!(
            Threshold::from_with_precision(0.99, 8).to_string(),
            "028f5c29"
        );
    }

    /// Inspired by tests in
    /// https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/pkg/sampling
    #[test]
    fn threshold_exhaustive() {
        struct Case {
            prob: f64,
            exact: &'static str,
            rounded: Vec<&'static str>,
        }

        // Note: the tests below express the threshold in terms of
        // (1 - hex_fraction), so the hexadecimal form of the fraction
        // equals the rejection threshold exactly. This tests for
        // correct rounding.
        let cases = [
            Case {
                prob: 1.0 - hexf64!("0x456789ap-28"),
                exact: "456789a",
                rounded: vec!["456789a", "45678a", "45679", "4568", "456", "45", "4"],
            },
            Case {
                prob: 1.0 - hexf64!("0xfff1f2e3d4c5p-48"),
                exact: "fff1f2e3d4c5",
                rounded: vec![
                    "fff1f2e3d4c5",
                    "fff1f2e3d4c",
                    "fff1f2e3d5",
                    "fff1f2e3d",
                    "fff1f2e4",
                    "fff1f2e",
                    "fff1f3",
                    "fff2",
                    "fff",
                    // Note: threshold will not lower precision at this
                    // point, because we lose orders of magnitude for very
                    // small probabilities..
                ],
            },
            Case {
                prob: 1.0 - hexf64!("0x0001f2e3d4c5bp-52"),
                exact: "0001f2e3d4c5b",
                rounded: vec![
                    "0001f2e3d4c5b",
                    "0001f2e3d4c6",
                    "0001f2e3d4c",
                    "0001f2e3d5",
                    "0001f2e3d",
                    "0001f2e4",
                    "0001f2e",
                    "0001f3",
                    "0001f",
                    "0002",
                    // Note: collapse to 100% sampling is allowed.
                    "0",
                ],
            },
        ];

        for k in cases {
            assert_eq!(
                Threshold::from_with_precision(k.prob, MAX_SAMPLING_PRECISION).to_string(),
                k.exact
            );

            // Ignore leading 'f', they are not counted in precision.
            let mut stripped = k.exact;
            while let Some(s) = stripped.strip_prefix("f") {
                stripped = s;
            }

            let leading_fs = k.exact.len() - stripped.len();

            for round in k.rounded {
                // Check that the input is not shorter than the number
                // of leading Fs, the test will fail b/c zero precision.
                assert!(round.len() >= leading_fs);

                let prec = (round.len() - leading_fs) as u32;

                let rth = Threshold::from_with_precision(k.prob, prec);

                assert_eq!(rth.to_string(), round);
            }
        }
    }

    #[test]
    fn trace_id_to_randomness() {
        let tid = opentelemetry::trace::TraceId::from_u128(0x101010101010101010123456789abcde);

        assert_eq!(
            Some(Randomness::from_trace_id(tid)),
            Randomness::from_rv_value("123456789abcde")
        );
    }
}
