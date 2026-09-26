# The BLE fitness data gap

*A teardown of what fitness and health devices actually expose, what shipping
products do with it, and where the unclaimed ground is.*

This document is the reasoning behind `raganything/biosignal/`. It is written to
be falsifiable: every gap below names the specific data that exists, the
specific thing products do with it instead, and the specific module here that
closes it.

---

## 1. The premise most products are built on is wrong

The industry's working assumption is that the constraint is **data access** —
that better health insight is gated on getting more sensors onto more people,
or on a partnership with a wearable vendor.

It isn't. The constraint is **what happens to data that has already been
collected**. A person wearing a mid-range chest strap and standing on a €40
smart scale is already broadcasting, over open royalty-free protocols, more
physiological detail than any consumer app on their phone displays or stores.
The bottleneck is a decade of product convention: collapse everything to one
number per day, discard the provenance, and never mention uncertainty.

That convention is the opportunity.

---

## 2. What the open standards actually put on the air

All of the following are Bluetooth SIG standard GATT services. No partner
agreement, no API key, no rate limit, no vendor review. Decoders for every one
of them live in [`raganything/biosignal/ble/codecs.py`](../raganything/biosignal/ble/codecs.py).

| Service | Characteristic | What it carries | Typically used for |
|---|---|---|---|
| Heart Rate (0x180D) | Heart Rate Measurement (0x2A37) | bpm, **RR intervals at 1/1024 s**, sensor-contact bit, energy expended | the bpm only |
| Cycling Power (0x1818) | CP Measurement (0x2A63) | watts, **pedal balance, accumulated torque, extreme force/torque magnitudes, top/bottom dead-spot angles** | watts only |
| Cycling Speed & Cadence (0x1816) | CSC Measurement (0x2A5B) | cumulative revolutions + device-clock event times | speed/cadence |
| Running Speed & Cadence (0x1814) | RSC Measurement (0x2A53) | speed, cadence, **stride length**, distance, walk/run state | pace |
| Fitness Machine (0x1826) | Indoor Bike / Treadmill / Rower (0x2AD2 / 0x2ACD / 0x2AD1) | power, cadence, **resistance level, incline, ramp angle, force on belt, MET, stroke rate** | speed, distance |
| Body Composition (0x181B) | BC Measurement (0x2A9C) | **raw bioimpedance in ohms**, plus the vendor's fat/muscle/water estimates | the fat percentage |
| Pulse Oximeter (0x1822) | PLX Continuous (0x2A5F) | SpO₂, pulse rate, **pulse amplitude index**, sensor status | SpO₂ |
| Glucose (0x1808) / CGM (0x181F) | 0x2A18 / 0x2AA7 | concentration, sample type and location, **sensor status annunciation** | the number |
| Battery (0x180F) | Battery Level (0x2A19) | percentage | nothing |

The bolded fields are the ones that essentially never reach a user-facing
surface. They are not exotic; they are in the same packet as the fields that do.

### The single biggest one: RR intervals

A Heart Rate Measurement notification can carry up to nine beat-to-beat
intervals at roughly 0.98 ms resolution
([Bluetooth SIG Heart Rate Service](https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/HRS_v1.0/out/en/index-en.html)).
This is the only raw physiological timing signal most people own outright. It is
the input to every HRV metric, to respiratory-rate estimation, to arrhythmia
screening, to autonomic load — and the overwhelming majority of apps read the
first byte, plot the bpm, and drop the rest of the packet on the floor.

---

## 3. The gaps, ranked

### Gap 1 — Provenance is destroyed at ingestion

**What happens today.** A measured heart rate from a chest strap, a heart rate
inferred by a wrist PPG algorithm, a heart rate relayed through a treadmill's
own smoothing, and an interpolated value filling a dropout all land in the same
column of the same table and render as the same line on the same chart. After
that point nothing downstream — no analytic, no model, no user — can distinguish
them.

**Why it matters.** Every conclusion inherits the weakest input silently. A
"recovery declining" trend built partly on interpolated segments looks exactly
like one built on clean measurement.

**Closed by.** [`schema.py`](../raganything/biosignal/schema.py). Every `Sample`
carries an `Evidence` class (`MEASURED` / `LOCALLY_DERIVED` / `VENDOR_DERIVED` /
`IMPUTED`) and a confidence weight; every `Stream` carries `Provenance` with the
device, transport, latency, the name of any closed algorithm involved, and
whether that derivation is publicly documented. A treadmill-relayed heart rate
is decoded as `VENDOR_DERIVED`, not `MEASURED`, because it is one. A smart
scale's impedance is `MEASURED` while the body-fat percentage computed from it
is not.

### Gap 2 — Missing data is drawn as a straight line and never mentioned again

**What happens today.** A sensor that loses contact for eleven minutes produces
a chart identical to one that recorded continuously, because the gap is
interpolated at render time. The interpolation is never labelled and never
propagates into the metrics computed over it.

**Why it matters.** This is the dominant error source in consumer physiology and
it is completely invisible. Wrist-worn HRV in particular is dominated by motion
artifact and missing data
([Error Estimation of Ultra-Short HRV Parameters](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7764161/)).

**Closed by.** [`analytics/quality.py`](../raganything/biosignal/analytics/quality.py).
Coverage is measured against the *session* window rather than the stream's own
extent, so a sensor that connected ten minutes late is caught rather than
looking perfect. Gaps, sampling jitter, decoder-flagged fraction, inferred
fraction, staleness and provenance completeness combine into one trust weight
with its reasons in plain language. `gate()` then suppresses any metric the
signal cannot support — and the refusal is as informative as a result would
have been.

### Gap 3 — HRV is reported as one number with none of the four facts that determine it

**What happens today.** One HRV figure per day, with no statement of the window
length, the beat count, the artifact-rejection rate, or the correction method.
Those four choices move the number more than any real day-to-day physiological
difference does. Validation work consistently finds that consumer devices track
heart rate far better than they track HRV
([Sinichi et al., *Psychophysiology* 2025](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.70004);
[Dial et al., *Physiological Reports* 2025](https://physoc.onlinelibrary.wiley.com/doi/10.14814/phy2.70527)),
and error varies by more than 10× with sensor contact pressure alone.

**Closed by.** [`analytics/hrv.py`](../raganything/biosignal/analytics/hrv.py).
Malik and Karlsson artifact correction, per-metric minimum windows (SDNN from a
30-second window is not the clinical SDNN and is not reported as if it were), a
hard artifact ceiling above which nothing is reported at all, and a transparent
confidence penalty product. One specific bug is fixed that inflates RMSSD
exactly on the noisiest nights: successive differences are never computed across
a removed beat, because that manufactures variability that did not occur.

### Gap 4 — Disagreement between devices is thrown away

**What happens today.** A person with a strap, a watch and a ring has three
heart rates. Platforms resolve this with a fixed priority list and show one.

**Why it matters.** The disagreement is often the most informative signal
available. Two devices agreeing to within 2 bpm all night and diverging by 25 bpm
during a lift have told you precisely which one to believe and when — a per-user,
per-context accuracy map that no vendor can produce for you because no vendor
sees the other devices.

**Closed by.** [`analytics/fusion.py`](../raganything/biosignal/analytics/fusion.py).
Selection by evidence class, then measured quality, then latency — with the
alternatives kept, compared Bland–Altman style (bias and 95% limits of
agreement), and any real conflict named explicitly in the record.

### Gap 5 — Derived fields nobody computes, from data everybody has

Standard packets already contain the inputs for analyses that are simply absent
from consumer products:

- **Aerobic decoupling** — power-per-heartbeat drift between the halves of an
  effort, the classic durability signal. Inputs: power + heart rate, both
  standard. Implemented in `analytics/load.py`.
- **Pedal-stroke asymmetry and dead-spot angles** — from the Cycling Power
  optional blocks, decoded and retained.
- **Cadence differentiated from cumulative counters** using the *device* clock
  rather than packet arrival, which removes radio jitter from the estimate —
  handled, including 16-bit event-time and counter rollover, by
  `RevolutionTracker`.
- **Raw bioimpedance trend** — a far better-behaved series than the vendor's
  body-fat regression sitting on top of it, and one whose within-device
  repeatability the user can actually assess.
- **Battery level as a data-quality covariate** — a strap at 4% is the single
  best predictor that the next hour of data is unreliable. It is trivially
  available and universally ignored.

### Gap 6 — Vendor scores are laundered into measurements

**What happens today.** A recovery or readiness score arrives from a cloud API
and is stored, charted and reasoned about identically to a sensor reading, with
its latency and its closed derivation both dropped.

**Why it matters.** These are real constraints, not pedantry: Fitbit's public
tier is rate-limited to 150 requests/hour with intraday access gated behind a
separate application; Garmin restates daily summaries retroactively, so the same
day fetched twice can differ; WHOOP reports per physiological cycle rather than
per calendar day, so aligning it to dates is itself an assumption; Oura's
readiness cannot be recomputed from anything Oura returns.

**Closed by.** [`sources.py`](../raganything/biosignal/sources.py). Declarative
per-vendor profiles that record real latency and granularity, mark scores
`VENDOR_DERIVED` with `documented=False` and the algorithm named, and never
promote them. Health Connect and HealthKit records are additionally split by
*writing app*, because those stores happily interleave a watch's step count and
the phone's pedometer into one series that looks continuous and is not.

### Gap 7 — The data has nowhere to be reasoned about

**What happens today.** Physiology lives in a time-series silo. The questions
people actually ask are relational and cross-domain — *does my sleep degrade in
the week after a hard block? which device disagrees during lifting? does what my
coach wrote match what my body did? what does the literature say about this
pattern in someone my age?* — and none of them are answerable by a time-series
query. Each becomes bespoke product code, so only the handful someone shipped
can ever be asked.

**Closed by.** [`narrative.py`](../raganything/biosignal/narrative.py) and
[`index.py`](../raganything/biosignal/index.py). Sessions are rendered as
evidence-annotated text and tables and inserted into RAG-Anything's knowledge
graph, where physiology sits alongside training plans, lab results, coach notes
and papers, and the joins are made by retrieval rather than by schema.

The property that makes this safe rather than reckless: the retrieved context
contains the caveats *in the same rows as the numbers*. Asked "why was my
recovery poor on the 14th?", the model retrieves not just the figure but the
fact that the strap lost contact for nine minutes and the RMSSD came from 41
beats. A model given that context qualifies its answer. A model given only the
number does not.

---

## 4. The architecture that follows

```
BLE GATT ──┐
           │   codecs.py          schema.py            analytics/         narrative.py
Vendor  ───┼──▶ decode bytes  ──▶ evidence-tagged  ──▶ gated metrics  ──▶ text + tables
cloud      │   (no radio dep)     streams              + quality          with caveats
           │                      + provenance         + fusion               │
Phone   ───┘                                                                  ▼
health store                                                        RAG-Anything graph
                                                                    (physiology beside
                                                                     plans, labs, papers)
```

Two deliberate properties:

- **The decoders have no Bluetooth dependency.** The hardest and most
  error-prone part of this domain — inverted flag bits, IEEE-11073 medical
  floats, wrapping counters — is pure functions over bytes, verified in CI
  against hand-assembled packets with no radio present. `bleak` is optional and
  imported lazily.
- **Refusal is a first-class output.** `withheld` sits beside `metrics`
  everywhere, carrying the reason. Athlete constants (resting HR, max HR,
  threshold power) have no defaults, because substituting a population estimate
  would let a score appear that is really a guess about the person wearing the
  device.

---

## 5. What this does *not* solve

Stated plainly, because a gap analysis that only lists winnable gaps is
marketing:

- **Sensor accuracy.** Nothing here improves a bad PPG signal. It measures and
  reports how bad it is, which is different and, given that contact pressure
  alone swings HRV error by more than an order of magnitude, arguably more
  useful — but it is not the same thing.
- **Vendor-gated data.** Sleep staging, intraday Fitbit history and several
  Garmin streams need partner approval. The design isolates that dependency
  rather than removing it.
- **Background collection on phones.** iOS restricts sustained BLE in the
  background, and Android battery optimisation varies by manufacturer. This is a
  platform-engineering problem, unaffected by anything in this package.
- **Regulated claims.** Everything here is analysis of consumer data. Screening
  or diagnostic claims are a regulatory undertaking, not a code change.

---

## 6. Where the defensible position is

Not in the sensors, which are commodity. Not in the analytics, which are
published. It is in being the only system in the category that **knows what it
does not know, says so, and can still answer questions across domains** —
because the provenance survives ingestion, the uncertainty survives the
analytics, and both survive into a knowledge graph a model can actually reason
over.

That combination is cheap to state and, judging by every shipping product in the
category, apparently very hard to give up the clean-looking chart for.

---

## Sources

- [Bluetooth SIG — Heart Rate Service](https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/HRS_v1.0/out/en/index-en.html)
- [Bluetooth SIG — Fitness Machine Service](https://www.bluetooth.com/specifications/specs/fitness-machine-service-1-0/)
- [Sinichi et al. (2025), *Psychophysiology* — accuracy of four heart-rate wearables](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.70004)
- [Dial et al. (2025), *Physiological Reports* — nocturnal RHR and HRV in consumer wearables](https://physoc.onlinelibrary.wiley.com/doi/10.14814/phy2.70527)
- [Error estimation of ultra-short HRV parameters under motion artifact](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7764161/)
- [Android Health Connect — data types](https://developer.android.com/health-and-fitness/health-connect/data-types)
- [Android Health Connect — reading raw data](https://developer.android.com/health-and-fitness/health-connect/read-data)
- [bleak — cross-platform BLE client for Python](https://bleak.readthedocs.io/)
