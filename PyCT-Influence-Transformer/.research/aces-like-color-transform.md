# ACES-like Color Transform Research

Date: 2026-09-08

Status: research and design proposal; no implementation yet.

## 1. Objective

The current brightness and contrast transforms operate directly on normalized
RGB channels. This is simple for symbolic execution, but it can change hue or
saturation when channels move by different amounts or when one channel is
clipped before the others.

The goal of this document is to define an ACES-like appearance transform that:

- adjusts tone through a shared lightness or luminance representation;
- preserves hue as much as possible;
- uses highlight and shadow roll-off instead of immediate hard clipping;
- compresses chroma when the result is outside the target RGB gamut;
- keeps a shared symbolic X controlling the transform;
- remains approximable by the current concolic execution architecture.

This is not intended to reproduce proprietary Adobe algorithms exactly. ACES
is used as the documented professional reference model.

## 2. Research findings

### Photoshop and Lightroom behavior

Adobe documents two important behaviors:

1. Legacy Brightness/Contrast shifts pixel values directly. Adobe warns that
   this can clip highlights or shadows and lose image detail.
2. Modern Brightness/Contrast uses proportionate, nonlinear adjustments. The
   newer Light workflow exposes exposure, contrast, highlights, shadows,
   whites, and blacks separately, and is non-destructive when used as an
   adjustment layer.

Adobe also distinguishes monochromatic contrast from per-channel contrast:

- applying the same contrast operation to all channels preserves the overall
  color relationship;
- expanding each channel independently can introduce or remove color casts.

Lightroom exposes clipping indicators because clipping can occur in only one
or two RGB channels. In that case the result is not necessarily white or
black; it can be a color-shifted highlight or shadow. If all channels clip,
the result becomes white or black and detail is lost.

Sources:

- [Photoshop Brightness/Contrast](https://helpx.adobe.com/uk/photoshop/using/apply-brightness-contrast-adjustment.html)
- [Photoshop quick tonal adjustments](https://helpx.adobe.com/photoshop/using/making-quick-tonal-adjustments.html)
- [Photoshop Light adjustment](https://helpx.adobe.com/photoshop/desktop/create-manage-layers/color-adjustment-fill-layers/adjust-image-lighting-with-light.html)
- [Lightroom tone curve](https://helpx.adobe.com/lightroom/mobile/adjust-light-and-color/adjust-tonal-range-using-curve.html)
- [Lightroom clipping behavior](https://helpx.adobe.com/ca/lightroom-classic/desktop/process-and-develop-photos/image-tone-color.html)

### ACES 2 behavior

ACES 2 uses an intermediate color-appearance representation rather than
applying the tone transform independently to display RGB channels. Its
documented rendering structure is approximately:

```text
ACES RGB
  -> JMh appearance representation
  -> tone scale on J (lightness)
  -> chroma compression on M (colorfulness)
  -> gamut compression
  -> display RGB
```

The purpose is to keep hue more stable, apply a softer highlight roll-off, and
avoid harsh clipping artifacts. ACES describes J as related to lightness, M as
colorfulness, and h as hue.

Sources:

- [ACES 2 output transforms](https://docs.acescentral.com/system-components/output-transforms/)
- [ACES tone mapping](https://docs.acescentral.com/system-components/output-transforms/technical-details/tone-mapping/)
- [ACES rendering transform](https://docs.acescentral.com/background/about-rendering/)

### Gamut mapping

After a tone operation, a color may still be outside the destination RGB
gamut. Hard per-channel clipping is the simplest mapping, but it can create
visible hue changes. The W3C CSS Color 4 specification describes a more
color-preserving strategy:

1. convert to a perceptual polar space such as OKLCh;
2. keep lightness and hue approximately constant;
3. reduce chroma until the color is inside the destination gamut;
4. convert back to RGB.

This sacrifices saturation before it sacrifices hue. It cannot preserve every
property simultaneously: if a color is outside the target gamut, some change
is unavoidable.

Source:

- [W3C CSS Color 4 gamut mapping](https://www.w3.org/TR/css-color-4/#gamut-mapping)

## 3. Current implementation and limitations

The current global-real builder is in
[`tasks/builders/global_real.py`](../PyCT-Influence-Transformer/tasks/builders/global_real.py).

Current coefficient definitions:

```text
brightness:
    coefficient[h,w,c] = 1

contrast:
    coefficient[h,w,c] = pixel[h,w,c] - mean_c

shap-sign:
    coefficient[h,w,c] = -1, 0, or +1
```

The runtime then applies the shared variable as:

```text
pixel_new = pixel_original + coefficient * X
```

In clip mode each RGB element is independently mapped to [0, 1]. The
implementation uses nested symbolic if-then-else expressions for this mapping
in [`libct/global_real.py`](../PyCT-Influence-Transformer/libct/global_real.py).

Important consequences:

- brightness uses all RGB channels and a common additive X, but independent
  clipping can still change color when only one channel reaches a boundary;
- contrast uses all RGB channels, but R, G, and B use different coefficients;
- contrast therefore changes channel relationships even before clipping;
- `clipped_count` counts out-of-range channel elements, not complete RGB
  pixels;
- the current symbolic model is affine per input element plus piecewise clip;
  it does not currently model a nonlinear color-space conversion.

The CIFAR10 input mapping covers every element of the image tensor through
`np.ndindex(sample.shape)`, producing one `v_h_w_c` variable for every
channel. There is no intentional channel omission.

## 4. Proposed ACES-like pipeline

The target conceptual pipeline is:

```text
normalized sRGB RGB
  -> linear RGB
  -> perceptual/appearance space
  -> tone operation on lightness
  -> highlight and shadow roll-off
  -> chroma compression if required
  -> gamut mapping to sRGB
  -> encoded RGB in [0, 1]
```

### 4.1 Input decoding

The dataset values are normalized sRGB values. For a color-accurate pipeline,
RGB should first be inverse-transfer-function decoded to linear RGB before
physical-light or luminance calculations.

This matters because adding values directly in encoded sRGB is not the same as
adding equal amounts of light. A first implementation should make the transfer
function explicit rather than silently treating encoded values as linear
light.

### 4.2 Intermediate representation

Two reasonable references exist:

#### ACES-like reference: JMh

Use an ACES-style JMh representation when the goal is to follow the professional
rendering architecture closely. Tone changes affect J, chroma changes affect M,
and hue is represented by h.

#### Engineering approximation: OKLCh

Use OKLCh when the goal is a documented, relatively compact perceptual space
with an explicit lightness, chroma, and hue decomposition. W3C's gamut-mapping
procedure is directly described in terms of OKLCh.

Recommendation: implement the architecture as ACES-like, but use OKLCh as the
first experimental representation unless the project specifically requires
ACES JMh compatibility.

### 4.3 Brightness control

Brightness should control the lightness component, not independently add the
same number to R, G, and B.

Conceptually:

```text
L_new = brightness_tone_curve(L, X)
C_new = C
h_new = h
```

An exposure-like control and a display-brightness control should be treated as
different variants:

- exposure-like: multiply or shift scene-referred light before tone mapping;
- display brightness: move the output tone curve while preserving the curve's
  toe and shoulder.

The experiment must choose one meaning for X and record it in metadata.

### 4.4 Contrast control

Contrast should modify the slope of a shared tone curve on lightness:

```text
L_new = contrast_tone_curve(L, X)
C_new = C
h_new = h
```

A useful curve has three regions:

- toe: protects dark detail;
- midtone region: controls the main contrast change;
- shoulder: protects highlight detail.

This is preferable to using one mean per RGB channel, because RGB channels do
not independently represent perceptual brightness.

### 4.5 Highlight roll-off and shadow toe

When the transform pushes highlights toward the upper boundary, the curve
should gradually reduce its slope rather than sending every over-range value
directly to 1. Similarly, the toe should gradually reduce the slope near 0.

Desired behavior:

```text
normal range:       approximately linear or contrast-controlled
near black:         smooth toe
near white:         smooth shoulder
outside target:     tone compression before gamut mapping
```

The exact curve can be a smooth curve in a reference implementation and a
piecewise-linear approximation in the symbolic implementation.

### 4.6 Chroma compression and gamut mapping

After changing lightness, convert the color back toward RGB and test whether
any component is outside [0, 1]. If it is outside:

```text
keep L approximately fixed
keep h fixed
decrease C
reconvert to RGB
repeat until in gamut
```

The final fallback may clamp RGB, but hard clipping should be the last output
step rather than the primary appearance operation.

## 5. Symbolic X design

The shared symbolic variable should remain one scalar X for the whole image.
It should control an appearance parameter, not act as an independent RGB
offset.

Conceptual brightness form:

```text
L_new = brightness_curve(L_original, X)
```

Conceptual contrast form:

```text
L_new = contrast_curve(L_original, X)
```

All pixels share the same X, while their lightness values determine where they
fall on the curve. The RGB output is then reconstructed from the transformed
lightness and the preserved color components.

### Exact symbolic implementation

An exact implementation would require symbolic support for some or all of:

- sRGB transfer functions;
- matrix color-space conversion;
- cube roots or other nonlinear functions;
- tone-curve nonlinearities;
- polar conversion and hue handling;
- iterative gamut mapping or binary search.

This is substantially beyond the current affine-plus-ITE global-real model.

### Piecewise-linear symbolic implementation

The practical option is to approximate the professional pipeline with a fixed
set of linear segments:

```text
if L is in segment 0:
    L_new = a0(X) * L + b0(X)
elif L is in segment 1:
    L_new = a1(X) * L + b1(X)
...
```

The same X remains shared. The symbolic expression becomes a bounded set of
ITE branches, which is compatible with the current execution model more than
general trigonometric or iterative expressions would be.

A second piecewise stage can approximate chroma compression:

```text
if RGB is inside gamut:
    keep color
else:
    reduce chroma using a bounded linear segment
```

The approximation must be validated against a concrete reference
implementation before it is used for concolic exploration.

## 6. Recommended implementation phases

### Phase 0: concrete reference transform

Before touching symbolic execution, implement a pure NumPy reference function
that accepts an image and X and returns the transformed image plus diagnostics.

Required diagnostics:

- original and transformed RGB values;
- lightness, chroma, and hue before and after;
- hue difference;
- chroma difference;
- luminance difference;
- number of hard-clipped channels;
- number of gamut-mapped channels or pixels.

This gives us a correctness oracle for later symbolic approximations.

### Phase 1: choose the representation and semantics

Decide explicitly:

- OKLCh or ACES JMh;
- brightness as exposure-like or display-brightness;
- contrast curve parameters;
- whether X is additive, multiplicative, or a curve-strength parameter;
- the target output gamut, initially sRGB [0, 1].

### Phase 2: piecewise approximation

Approximate the selected tone curve and gamut compression using a small fixed
number of linear pieces. Measure approximation error against the reference
transform over representative CIFAR10 images and X values.

### Phase 3: symbolic integration

Extend `global_real` only after the reference and approximation contracts are
stable. Preserve the existing shared-X contract and record the transform
version and parameters in `global_real_config`.

### Phase 4: experiment comparison

Compare the existing RGB-affine attack with the ACES-like attack using the same:

- cases;
- X bounds;
- solver timeout;
- model;
- success criterion.

Report both attack success and color-fidelity diagnostics. A higher success
rate alone is not sufficient if the attack relies on severe gamut distortion.

## 7. Acceptance criteria

The ACES-like transform should not be considered successful only because it
changes the model label. It should also satisfy:

- hue drift is lower than the current per-channel contrast baseline;
- hard clipping is reduced or clearly isolated to unavoidable output cases;
- highlight detail is not collapsed by a single-channel boundary hit;
- the concrete reference and piecewise approximation agree within a defined
  error tolerance;
- symbolic X remains one shared solver variable;
- metadata records color-space, curve, gamut-mapping, and approximation
  parameters;
- failure to map a color is explicit rather than silently treated as a normal
  affine result.

## 8. Open decisions for implementation discussion

1. Use OKLCh first, or implement an ACES JMh-compatible path directly?
2. Should brightness X represent exposure, display brightness, or tone-curve
   offset?
3. Should contrast use a fixed curve family or expose toe/midtone/shoulder
   parameters?
4. How many linear segments can the solver handle before timeouts become
   unacceptable?
5. Should gamut mapping prefer constant hue and lightness with chroma
   reduction, or allow a small lightness change to preserve more chroma?
6. Which color-fidelity metrics should be persisted in `stats.json`?

## 9. Scope boundary

This document is research and design documentation only. It does not change
the current brightness, contrast, SHAP-sign, symbolic execution, or experiment
code.
