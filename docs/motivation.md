# 从检索增强分类到比较式开放词汇农业诊断

Status: paper-facing research motivation and proposed method framing. This
document does not claim that the proposed framework, its enlarged knowledge
base, or its training data have already been implemented or evaluated.

## One-sentence thesis

Open-vocabulary agricultural disease and pest recognition is fundamentally a
**comparative diagnosis** problem: a system should use visual observations to
form competing, confusable hypotheses and retrieve external knowledge that can
*distinguish* those hypotheses, rather than retrieve a plausible answer and
let it overwrite the image evidence.

## 1. Why conventional classification and RAG are insufficient

Agricultural disease and pest recognition has three coupled properties.

1. **Open vocabulary and long tail.** Test images can contain categories,
   aliases, language forms, or fine-grained variants that are not adequately
   represented in parametric training data.
2. **Fine-grained visual confusability.** Closely related diseases and pests
   often share host, colour, lesion shape, or body appearance; the decisive
   cue may be subtle, local, or stage-dependent.
3. **Plausible retrieval can be harmful.** A visually or semantically similar
   but incorrect retrieved class is persuasive enough to pull a vision-language
   model away from an initially useful Direct prediction.

Consequently, the task is not adequately captured by either closed-set image
classification or the usual "image/question -> retrieve documents -> generate
an answer" RAG pipeline. The critical question is not only *which class is
relevant?* It is *which observable and publicly verifiable evidence separates
the remaining similar classes?*

We therefore formulate the task as **comparative hypothesis verification for
open-vocabulary agricultural recognition**. A prediction is the outcome of a
diagnostic comparison among plausible alternatives, not merely the highest
scoring class name.

## 2. Proposed paradigm: Hypothesize--Contrast--Verify

We propose the paper-level paradigm **Hypothesize--Contrast--Verify (HCV)**,
also described as **Comparison-Centric Retrieval-Augmented Recognition**. It
reorganizes the roles of Direct visual recognition and retrieval:

```text
Agricultural image
        |
        v
Visual hypothesis formation
  observations + confusable candidate classes + uncertainty
        |
        v
Contrast state
  which candidate differences remain unresolved?
        |
        v
Similar-class knowledge retrieval
  retrieve discriminative attributes, aliases, host/symptom evidence,
  and reference evidence for the unresolved comparison
        |
        v
Visual--knowledge verification
  support, rule out, or revise competing hypotheses
        |
        v
Open-vocabulary agricultural diagnosis
```

The Direct component is therefore not a baseline that RAG replaces. It is the
visual diagnostic anchor that identifies what must be compared. RAG is not a
generic knowledge appendage; it is an instrument for resolving a specific
unsettled candidate relation.

### The contrast state

The central intermediate representation is a compact, inspectable **contrast
state**, not an unconstrained verbose chain of thought. It records:

- observable visual attributes (host, affected organ, lesion or body shape,
  colour, texture, distribution, developmental stage);
- a small set of visually confusable class hypotheses;
- discriminative cues that would separate those hypotheses;
- evidence that is missing, contradictory, or insufficient; and
- a retrieval question targeted at resolving that particular contrast.

For example, rather than searching only for a presumed label, a model can state
that it observes rough black structures on a potato surface, contrast
black-scurf-like symptoms with leaf-focused blights, and retrieve the public
features that distinguish lesion location and characteristic structures. The
final decision must then be consistent with both the visible image evidence and
the retrieved differential evidence.

This makes the learned reasoning object a **discriminative contrastive
reasoning trace**: a concise, checkable diagnostic process centred on candidate
differences, rather than a free-form explanation or an imitation of a teacher's
private reasoning.

## 3. What makes HCV different from conventional RAG

The distinction is architectural and epistemic, not simply an additional tool
call.

| Conventional multimodal RAG | HCV comparative recognition |
| --- | --- |
| Image/question -> retrieve relevant material -> generate an answer | Image -> form confusable hypotheses -> retrieve evidence that distinguishes them -> verify a decision |
| The retrieval target is a relevant answer or document | The retrieval target is a discriminative fact, alias, or reference relation |
| Image evidence and retrieved context are often concatenated as peer inputs | Direct visual observations remain a comparison anchor against which retrieved evidence is checked |
| Additional turns mainly add more context | An additional turn is justified only by an unresolved candidate contrast |
| Success is predominantly retrieval relevance and final accuracy | Success also requires candidate coverage, discriminative value, visual--knowledge consistency, and auditable resolution |

This framing provides a principled answer to the observed failure mode in
multimodal RAG: retrieval is not automatically beneficial merely because its
content is relevant. In fine-grained agriculture, a similar-class result can
be relevant yet diagnostically misleading. HCV requires external evidence to
participate in a comparison with the Direct visual prior, rather than allowing
the retrieved top result to become an unexamined replacement answer.

### Relationship to existing adaptive RAG

Existing systems such as Self-RAG, Adaptive-RAG, and FLARE primarily study when
to retrieve and how much retrieval is needed. CRAG evaluates retrieved
evidence and triggers correction when it is unreliable. Query rewriting seeks
better formulations for retrieval. Multimodal reliability work, including
RULE, further shows that retrieved content can cause a vision-language model to
abandon a correct initial prediction.

HCV is complementary but asks a different question: **given a visual
fine-grained ambiguity, what relation among candidate classes should the model
retrieve in order to resolve that ambiguity?** Its queries are not generic
rewrites of the user request; they arise from a candidate pair or set and from
the missing diagnostic cue that separates them. The comparison anchor also
makes retrieval correction bidirectional: public knowledge can revise a visual
hypothesis, but visual evidence can flag a retrieved candidate as insufficient
or inconsistent.

The claim should remain appropriately scoped. The contribution is not the
invention of multi-hop RAG, adaptive retrieval, or comparison itself. The
proposed novelty is their integration into a **comparison-centred diagnostic
paradigm for open-vocabulary, long-tail agricultural recognition**, where
similar-class confusion is the primary object of retrieval and decision.

## 4. Knowledge-base contribution: from corpus to comparative evidence space

The resource should be positioned as an **Open-Vocabulary Agricultural
Visual--Semantic Differential Knowledge Base**, rather than as a generic
document collection. Its purpose is to make class differences publicly
retrievable and auditable.

The currently supported foundation includes canonical bilingual class names,
aliases, reference images, visual, text, fused, and name-oriented search
modes, as well as curated same-domain similar-class lists. A paper-facing
expansion should represent, for each class and where available, the following
comparable evidence:

- canonical names and multilingual/common aliases;
- reference images and visual retrieval representations;
- host, affected part, morphology, symptom, and developmental-stage
  descriptions;
- cross-modal indices for visual, semantic, fused, and name lookup;
- links to visually confusable categories; and
- discriminative attributes or diagnostic cues that distinguish those
  categories.

The first five elements describe the current database direction, including the
existing curated similar-class relations. Explicit per-pair discriminative
attributes remain a proposed enrichment and must not be claimed as completed
until constructed and validated. Together, they transform the resource from a
store of class descriptions into a **retrievable comparison space**: the
system can ask not only "what resembles this image?" but also "what
observable property separates these two public candidates?"

This is particularly valuable for open-vocabulary recognition: aliases connect
different naming conventions, reference evidence grounds visual similarity, and
differential attributes let the system explain why one long-tail class is
preferred over its plausible neighbours.

## 5. Learning the paradigm without making distillation the headline

The framework needs supervision, but teacher distillation should be described
as an **instantiation mechanism**, not the paper's primary novelty. The target
of learning is not a teacher's unrestricted internal reasoning. It is the
observable, auditable sequence required by HCV:

1. identify visible diagnostic attributes;
2. form a small, uncertainty-aware set of confusable hypotheses;
3. state the candidate difference that is still unresolved;
4. formulate a public query for discriminative evidence;
5. compare retrieved evidence with image observations; and
6. support, reject, or revise the hypotheses before producing the final class.

This changes the role of a multi-turn trace. A second query is not evidence of
better reasoning merely because it exists. It is useful only if it addresses an
unresolved contrast and changes the public evidence available for a decision.
Candidate-name confirmation, semantic attribute search, visual refinement, and
fused retrieval can all be concrete actions inside the same HCV loop.

Training-data construction can therefore enforce **evidence-delta** and
public-evidence constraints: retain traces in which a follow-up exposes a new
candidate, discriminative attribute, alias resolution, or meaningful rank
change; retain valid one-query stop decisions; and reject duplicate or
no-value tool trajectories. These are quality controls for learning the
comparative diagnostic process, not independent headline claims.

## 6. Paper-facing contributions

The intended contribution structure is:

1. **Problem formulation.** Formulate open-vocabulary agricultural disease and
   pest recognition as comparative hypothesis verification under long-tail
   visual confusability, rather than as closed-set classification or answer
   retrieval.
2. **Framework.** Introduce HCV, a Direct-anchored, comparison-centric
   retrieval-augmented recognition framework in which retrieval seeks evidence
   that distinguishes visually confusable agricultural candidates.
3. **Reasoning object.** Define a compact discriminative contrastive reasoning
   trace that connects visual observations, competing hypotheses, unresolved
   diagnostic cues, public retrieval, and evidence-based revision.
4. **Resource.** Build and evaluate an open-vocabulary agricultural
   visual--semantic differential knowledge base that supports class, alias,
   image, attribute, and similar-class comparison.
5. **Evaluation.** Establish that improvements arise from more useful
   comparative evidence, not simply from more retrieved tokens or more tool
   calls, using strict protocol evaluation and paired analyses.

An NMI-style central claim can be stated as follows:

> We recast open-vocabulary agricultural recognition as comparative diagnosis.
> Rather than retrieving a plausible class description after visual prediction,
> our framework uses visual observations to construct confusable hypotheses and
> retrieves public evidence specifically to distinguish them. This creates an
> auditable visual--knowledge verification loop for long-tailed disease and
> pest recognition.

## 7. Evidence boundary and empirical agenda

The present evidence motivates the problem but does not yet validate HCV:

- **M1** is a strong Direct reference on the current 618-image bridge
  (67.31% Direct), showing that parametric visual recognition can be a valuable
  anchor.
- **M3** is the protocol-clean strict-RAG reference: checkpoint-72 reaches
  60.52% versus its matched strict raw base at 54.85%, a paired +5.66pp
  improvement (95% CI [+3.07, +8.25]).
- The M3/epoch-6 trajectory audit identifies two distinct limitations: truth
  candidates are often absent from public retrieval, and a model can fail to
  use retrieved truth-compatible evidence. The difficult M1-gap slices include
  known Option, known pest, unknown Open, and unknown disease.

These findings support the motivation for comparative retrieval but must not be
presented as a completed validation of the new paradigm. The appropriate
experimental sequence is:

| Comparison | What it isolates |
| --- | --- |
| Current M3 one-query strict RAG | Protocol-clean starting point |
| Fixed two-query RAG | Whether call count alone helps |
| Query rewriting without contrast state | Generic retrieval reformulation |
| HCV without selective trace filtering | Value of the Direct-anchored comparison loop |
| Full HCV | Combined value of comparison-guided retrieval and high-value traces |

Each route should report overall accuracy and independent slices for
known/unknown, Open/Option, and disease/pest. It should additionally report
first- and final-turn truth-hit@k, accuracy conditional on a truth hit, mean
tool calls, retrieval-mode distribution, evidence-change rate,
visual--knowledge conflict/resolution statistics, and strict protocol-error
rates. Matched raw-base paired bootstrap remains the causal RAG-effect
comparison; M1 Direct is a deployment reference, not a matched RAG baseline.

## 8. Practical design principles

- Keep Direct visual reasoning as an anchor, never as disposable preamble.
- Retrieve to separate plausible candidates, not merely to repeat a predicted
  label.
- Require every extra retrieval turn to name the unresolved comparison it is
  intended to settle.
- Preserve public evidence lineage: a follow-up name or alias must originate
  in earlier public retrieval, never from a hidden ground-truth label.
- Measure whether retrieval changes usable evidence, not only whether it
  returns more text.
- Fail closed on invalid tool behaviour and report it separately from task
  accuracy.
- Treat database coverage and evidence use as separate bottlenecks: improving
  one does not prove that the other is solved.

## 9. Terminology to keep stable

| Concept | Recommended paper term |
| --- | --- |
| Overall paradigm | Hypothesize--Contrast--Verify (HCV) / Comparison-Centric Retrieval-Augmented Recognition |
| Task view | Comparative hypothesis verification for open-vocabulary agricultural recognition |
| Direct intermediate state | Visual hypothesis or Direct visual prior |
| Structured comparison representation | Contrast state |
| Learnable trace | Discriminative contrastive reasoning trace |
| Database | Open-Vocabulary Agricultural Visual--Semantic Differential Knowledge Base |
| Retrieval purpose | Similar-class discriminative retrieval |
| Final mechanism | Visual--knowledge verification loop |

This vocabulary keeps the paper's centre of gravity on a new diagnostic
paradigm and its supporting knowledge infrastructure. Data construction,
teacher generation, multi-query scheduling, and evidence-delta filtering are
important implementation choices, but they should serve this central story.
