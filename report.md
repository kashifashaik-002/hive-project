# AppleSupport AI Agent — Hiver SDE Intern Take-Home

## 1. Problem Framing

### Objective

The goal of this project is to build a lightweight customer-support agent for AppleSupport using historical customer-support conversations from the Customer Support on Twitter dataset.

For an incoming customer message, the system performs three tasks:

1. **Intent classification** — identify the primary support issue.
2. **Historical grounding** — retrieve similar historical customer cases and their AppleSupport responses.
3. **Handling decision** — decide whether the case can be auto-handled or should be escalated.

I selected AppleSupport because the dataset contains a large number of historical support interactions covering software, hardware, account, connectivity, billing, and service-related problems.

### What does "good" mean?

For this use case, a good support agent should:

* correctly identify the customer's primary issue;
* retrieve historically relevant AppleSupport cases;
* avoid inventing unsupported troubleshooting information;
* provide a clear and relevant response;
* escalate sensitive or operationally complex cases;
* make escalation decisions explainable;
* remain useful when messages contain conversational noise or multiple issues.

I therefore evaluate the system across three main dimensions:

**intent correctness + historical grounding + safe escalation.**

---

## 2. Intent Taxonomy

I manually defined eight primary intents:

| Intent                    | Description                                                                    |
| ------------------------- | ------------------------------------------------------------------------------ |
| `battery_performance`     | Battery drain, charging performance, and battery health                        |
| `ios_software_bug`        | iOS/macOS bugs, crashes, freezing, lag, and software malfunction               |
| `update_installation`     | Problems installing or updating iOS, apps, or software                         |
| `hardware_device`         | Physical devices, accessories, screens, keyboards, cables, and device behavior |
| `connectivity_network`    | Wi-Fi, Bluetooth, cellular, and network connectivity                           |
| `account_icloud_security` | Apple ID, iCloud, passwords, and suspicious account/security activity          |
| `apple_apps_services`     | Apple Music, Maps, App Store, Weather, and other Apple services/apps           |
| `billing_purchase_repair` | Purchases, payments, refunds, billing, repair, and replacement                 |

For multi-issue messages, I use a **single primary-intent policy**, selecting the issue that appears to be the main reason for contacting support.

This makes the classification task deterministic and easier to evaluate, although it is also a limitation for messages that genuinely contain multiple independent problems.

---

## 3. What I Chose Not to Build

I deliberately kept the system lightweight and reproducible.

I did not build:

* an LLM fine-tuning pipeline;
* a multi-turn conversation memory system;
* live Apple knowledge-base integration;
* automatic account/device actions;
* a production-grade human handoff system;
* a large neural retrieval model;
* a fully autonomous tool-calling agent.

The goal was to demonstrate an end-to-end support pipeline with interpretable components rather than maximize model complexity.

---

# 4. Dataset and Golden Evaluation Set

I used the **Customer Support on Twitter** dataset and filtered it to AppleSupport interactions.

The original dataset contains customer messages, brand responses, timestamps, tweet IDs, and response relationships.

For AppleSupport, customer messages were connected to AppleSupport responses using the customer's `response_tweet_id` and the corresponding AppleSupport response `tweet_id`.

### Golden set

The final golden evaluation set contains **200 manually labelled examples**, which is within the required 150–250 example range.

Each example contains:

* tweet ID;
* customer message;
* cleaned message;
* manually assigned intent;
* manually assigned escalation decision;
* escalation reason.

### Sampling and labelling

The examples were sampled from AppleSupport customer interactions rather than generated synthetically.

I manually reviewed the selected examples and assigned:

1. one primary intent from the eight-intent taxonomy;
2. an escalation decision;
3. an escalation reason where appropriate.

Ambiguous and multi-issue messages were labelled according to the most prominent customer problem.

The full raw Twitter dataset is not included in the repository.

---

# 5. System Design

The pipeline consists of four main components.

```text
Customer Message
       |
       v
Intent Classifier
       |
       +----------------------+
       |                      |
       v                      v
Historical Retrieval    Escalation Rules
       |                      |
       +----------+-----------+
                  |
                  v
          Response Generator
                  |
                  v
        Final Support Response
```

## 5.1 Intent Classification

I used a TF-IDF representation followed by Logistic Regression.

Configuration:

* TF-IDF features;
* English stop-word removal;
* unigram and bigram features;
* `min_df=2`;
* maximum 20,000 features;
* Logistic Regression;
* `class_weight="balanced"`;
* `max_iter=1000`.

The classifier was evaluated using **5-fold stratified cross-validation** to avoid evaluating predictions from a model trained directly on the same examples.

---

## 5.2 Historical Retrieval

Historical customer messages are represented using TF-IDF.

For an incoming message:

1. clean the message;
2. transform it into the TF-IDF space;
3. calculate cosine similarity against historical customer messages;
4. retrieve the top three similar cases.

Each retrieved case contains:

* the historical customer message;
* the historical AppleSupport response.

The retrieved response is used as historical grounding evidence.

---

## 5.3 Escalation

Escalation is implemented as a transparent rule-based component.

The current rules cover categories such as:

* billing;
* account/security;
* repair/replacement;
* severe or unsafe situations.

Example trigger groups include:

```text
billing:
charged, charge, refund, money, payment, purchase

security:
hacked, stolen, phishing, scam, suspicious, account compromised

repair:
repair, replacement, replace, service center, broken

severe:
emergency, danger, unsafe, fire, smoke
```

If an escalation trigger is detected, the system returns:

```text
ESCALATE
```

along with the matched reason.

Otherwise it returns:

```text
AUTO-HANDLE
```

with a routine-support explanation.

This design was chosen because the decision is transparent and easy to audit.

---

## 5.4 Response Generation

For auto-handled cases, the system retrieves the strongest historical match and uses the corresponding AppleSupport response as grounding evidence.

Handles and URLs are removed before the historical response is incorporated into the final response.

The response is then wrapped in a lightweight template that identifies the detected issue and provides a next-step fallback.

For escalation cases, the system produces a conservative handoff response instead of attempting to solve potentially sensitive cases autonomously.

The response generator is intentionally lightweight. It is **not an LLM**; it primarily reuses and lightly transforms historical support responses.

---

# 6. Evaluation Results

The system was evaluated on:

1. intent classification;
2. escalation;
3. historical retrieval;
4. response quality.

## 6.1 Intent Classification

The classifier was evaluated using 5-fold stratified cross-validation.

| Metric          |            Result |
| --------------- | ----------------: |
| Intent Accuracy | **0.285 (28.5%)** |
| Intent Macro-F1 |        **0.2402** |

The results show that the current TF-IDF + Logistic Regression classifier struggles with the manually defined intent boundaries.

This is particularly visible for semantically overlapping categories such as:

* software bug vs update installation;
* hardware vs software;
* Apple services vs connectivity;
* billing vs Apple services.

The relatively low accuracy is therefore an important finding rather than something hidden by reporting only the retrieval results.

---

# 7. Baseline Comparison

The assignment requires comparison with both a trivial baseline and a simple baseline.

## 7.1 Trivial Baseline

The trivial baseline predicts the majority intent for every incoming message.

Its purpose is to establish the performance obtainable without learning meaningful message-specific patterns.

**Majority-class baseline result: TBD — to be calculated from the golden set.**

---

## 7.2 Simple Keyword Baseline

The simple baseline uses keyword groups corresponding to the eight intent categories.

For example:

```text
battery:
battery, charging, charge, drain

connectivity:
wifi, wi-fi, bluetooth, network, signal

update:
update, updating, install, upgrade

account:
icloud, apple id, password, login, hacked

billing:
charged, payment, refund, purchase

hardware:
screen, keyboard, cable, device, iphone, ipad, macbook
```

This provides a simple non-statistical comparison against the TF-IDF + Logistic Regression classifier.

**Keyword-baseline result: TBD — to be calculated from the golden set.**

### Current comparison

| Approach                     |  Accuracy |   Macro-F1 |
| ---------------------------- | --------: | ---------: |
| Majority-class baseline      |   **TBD** |    **TBD** |
| Keyword baseline             |   **TBD** |    **TBD** |
| TF-IDF + Logistic Regression | **28.5%** | **0.2402** |

The baseline values will be filled in after running the corresponding baseline evaluation.

---

# 8. Escalation Evaluation

The positive class is `ESCALATE`.

| Metric    |     Result |
| --------- | ---------: |
| Precision | **0.3810** |
| Recall    | **0.2286** |
| F1        | **0.2857** |

The relatively low recall indicates that the current keyword-based rules miss a number of escalation-worthy cases.

The low precision also indicates that some ordinary customer complaints contain words that trigger escalation even when human review is not necessarily required.

This shows that escalation requires more contextual understanding than simple keyword matching.

---

# 9. Historical Retrieval Evaluation

For every golden-set message, the system retrieved the top three historical cases.

The primary retrieval diagnostic was the similarity of the strongest retrieved case.

| Retrieval Metric             |     Result |
| ---------------------------- | ---------: |
| Mean Top-1 Cosine Similarity | **0.8864** |

The high average similarity indicates strong lexical overlap between customer messages and retrieved historical cases.

However, cosine similarity is **not equivalent to answer correctness**.

A retrieved case can have similar words while representing a different underlying problem. Therefore, retrieval similarity is treated as a diagnostic rather than a complete quality metric.

---

# 10. Response Evaluation and LLM-as-Judge

The evaluation harness defines an LLM-as-judge rubric for generated responses.

Each response is evaluated on five dimensions:

1. **Relevance**
2. **Helpfulness**
3. **Groundedness**
4. **Safety**
5. **Clarity**

Each dimension is scored from 1–5.

The judge also produces:

* overall score;
* short reason.

The judge receives:

* customer message;
* predicted intent;
* escalation decision;
* generated response;
* historical evidence.

This allows the judge to assess whether the response is supported by the retrieved evidence.

### Important limitation

The current implementation includes the **LLM-as-judge prompt/harness**, but I did not fabricate LLM scores that were not actually run.

Therefore, the repository does not claim unverified response-quality scores.

### Human agreement

A complete production evaluation would have a human independently score a sample of responses using the same rubric and compare the results with the LLM judge.

Possible agreement measurements include:

* score correlation;
* exact or adjacent-score agreement;
* weighted Cohen's kappa after converting scores into categories.

This was identified as a remaining evaluation gap.

---

# 11. Failure Analysis

The following examples are actual errors observed in the evaluation output.

## Failure Mode 1 — Update Installation vs General Software Bug

### Example

> "I hit update and it acts like it will work and then don’t. It’s working again but updating apps really slow."

**True intent:** `update_installation`

**Predicted intent:** `ios_software_bug`

### Hypothesis

The message contains both update-specific language and general malfunction language.

Terms such as "slow" and "working" overlap with general software-failure examples, so the lexical classifier can focus on the symptom instead of the operation causing the problem.

### Potential improvement

Use semantic representations and explicitly model actions such as:

* updating;
* installing;
* downloading;

separately from symptoms such as:

* crashing;
* freezing;
* lagging.

---

## Failure Mode 2 — Hardware vs Software

### Example

> "My Smart Keyboard keeps disconnecting from my iPad Pro (10.5). Like 2-3 per day. Restarting is the only sure way to fix it."

**True intent:** `hardware_device`

**Predicted intent:** `ios_software_bug`

### Hypothesis

The message is about a physical accessory, but the fact that restarting fixes the problem introduces software-related vocabulary.

The classifier therefore associates the message with software troubleshooting instead of identifying the keyboard as the primary object.

### Potential improvement

Add entity-aware features that distinguish physical accessories from operating-system problems.

---

## Failure Mode 3 — Apple Service vs Connectivity

### Example

> "Apple Maps on #CarPlay also doesn’t work in the #UAE. And rather than taking the feedback, the support team give a ‘helpful’ link."

**True intent:** `apple_apps_services`

**Predicted intent:** `connectivity_network`

### Hypothesis

The message describes an Apple service that does not work in a particular context.

The classifier appears to associate "doesn't work" and CarPlay-related context with connectivity even though the main subject is Apple Maps.

### Potential improvement

Give greater importance to explicitly named Apple products/services when the actual complaint is about their functionality.

---

## Failure Mode 4 — Apple Services vs Billing

### Example

> "I’m not subscribed to Apple Music. I can play it from the app."

**True intent:** `apple_apps_services`

**Predicted intent:** `billing_purchase_repair`

### Hypothesis

The phrase "not subscribed" resembles subscription and payment-related examples.

However, the customer is describing Apple Music functionality rather than a financial transaction.

This demonstrates that lexical features can over-weight individual words without fully understanding their context.

### Potential improvement

Use semantic classification and require stronger financial evidence before assigning the billing category.

---

## Failure Mode 5 — Multi-Issue Software Messages

### Example

> "Yo @AppleSupport 11.0.2 is fucking shocking. Laggy, buggy asf. Notifications hit and miss. Sort it!"

**True intent:** `ios_software_bug`

**Predicted intent:** `apple_apps_services`

### Hypothesis

The message contains multiple symptoms:

* lag;
* bugs;
* notification failures.

The classifier must still select one intent.

This demonstrates a limitation of the single-label taxonomy.

### Potential improvement

A future version could use:

* multi-label classification;
* hierarchical classification;
* primary-issue extraction before intent classification.

---

# 12. Escalation Failure Analysis

The escalation rules produced both false positives and false negatives.

## False Positive — Repair Keyword

Example:

> "No, thank you. If it involves spending 2 hours at a Genius Bar and spending money on repair, I’ll pass."

**True:** `AUTO-HANDLE`

**Predicted:** `ESCALATE`

### Hypothesis

The word "repair" directly triggered the repair rule even though the customer was declining repair rather than requesting additional assistance.

### Lesson

Keyword presence does not necessarily indicate customer intent.

---

## False Positive — Customer Frustration

Example:

> "This is just so disappointing. I really love my 6s plus... this really forces me to consider other brands."

**True:** `AUTO-HANDLE`

**Predicted:** `ESCALATE`

### Hypothesis

The system over-interprets a strongly negative complaint even though there is no explicit escalation trigger.

### Lesson

Customer frustration should not automatically be interpreted as an escalation requirement.

---

## False Negative — Account Security

Example:

> "Uhhhh @AppleSupport, this you? why is it saying my Apple ID will be disabled?"

**True:** `ESCALATE`

**Predicted:** `AUTO-HANDLE`

### Hypothesis

The message describes a potentially security-sensitive Apple ID issue but does not contain one of the explicit security keywords used by the current rule set.

### Lesson

Security-related escalation requires contextual understanding rather than only explicit words such as "hacked" or "phishing."

---

## False Negative — Complex Device Issue

Example:

> "My battery is struggling to hold its charge, the screen keeps freezing, apps are crashing or just stop working properly...."

**True:** `AUTO-HANDLE`

**Predicted:** `ESCALATE`

### Hypothesis

Multiple symptoms make the rule-based system overly conservative.

### Lesson

Escalation should be based on the type of action required, not simply the number or severity of symptoms mentioned.

---

# 13. What Is Misleading About My Headline Number?

The headline numbers should be interpreted carefully.

## 1. The golden set contains only 200 examples

Although 200 examples satisfy the assignment requirement, this is still a small sample relative to the original dataset.

The results should therefore not be interpreted as production-level performance.

## 2. The intent taxonomy is manually defined

The eight categories were designed for this project.

Different reasonable taxonomies could produce different classification results.

For example, an Apple Maps problem involving CarPlay could reasonably be considered either a service problem or a connectivity problem depending on the exact failure.

## 3. Accuracy hides category-level difficulty

The overall intent accuracy is **28.5%**, but this single number does not explain which categories are difficult.

The actual errors show that confusion is concentrated around semantic boundaries such as:

* update vs software bug;
* hardware vs software;
* service vs connectivity;
* service vs billing.

Therefore, the confusion matrix and individual examples are more informative than accuracy alone.

## 4. Retrieval similarity is not response correctness

The mean top-1 similarity is **0.8864**, which appears strong.

However, this is a lexical similarity measure.

A high score does not prove that the retrieved historical response is appropriate for the current customer.

## 5. Intent accuracy does not measure response quality

A correct intent prediction does not guarantee that the final response is:

* helpful;
* safe;
* grounded;
* clear.

This is why the project separates intent, retrieval, escalation, and response evaluation.

## 6. Escalation has asymmetric consequences

The escalation recall is **0.2286**, meaning the current rule-based approach misses a substantial portion of the labelled escalation cases.

This is important because a missed escalation can matter more than an unnecessary escalation for security or operationally sensitive cases.

## 7. The response generator is deliberately lightweight

The response generator is not a general-purpose LLM.

It primarily reuses a historical AppleSupport response and places it inside a response template.

Therefore, the intent result should not be interpreted as evidence that the complete system can autonomously solve arbitrary support conversations.

---

# 14. What I Would Build With One More Week

With another week, I would focus on improving reliability rather than adding complexity for its own sake.

## 1. Semantic retrieval

Replace TF-IDF retrieval with sentence embeddings.

This would allow semantically similar messages with different wording to retrieve each other.

For example:

```text
"battery dies very quickly"
```

and

```text
"my phone won't hold a charge"
```

could be recognized as similar even without strong lexical overlap.

---

## 2. Stronger intent classifier

I would replace the current TF-IDF classifier with a semantic classifier or embedding-based approach.

I would also consider hierarchical classification:

```text
Software
 ├── Software bug
 └── Update problem

Hardware
 ├── Device
 └── Accessory

Account
 ├── Apple ID
 └── Security
```

This could reduce confusion between closely related categories.

---

## 3. Better escalation model

I would replace the keyword-only approach with a hybrid:

```text
Rules + semantic classifier + confidence threshold
```

Security-sensitive categories would use conservative routing.

---

## 4. Confidence-aware routing

The system should estimate confidence for:

* intent;
* retrieval;
* escalation.

Low-confidence cases could automatically move to human review rather than forcing an uncertain automated response.

---

## 5. LLM response synthesis

Instead of directly reusing the strongest historical response, an LLM could synthesize a response from the top retrieved cases.

The prompt would require the model to:

* use only retrieved evidence;
* avoid inventing policies;
* avoid fabricating troubleshooting steps;
* preserve AppleSupport's historical support style;
* escalate when evidence is insufficient.

---

## 6. Complete response evaluation

I would complete the LLM-as-judge evaluation with a human-labelled subset.

The same rubric would be used by:

* the automated judge;
* human reviewers.

The results could then be compared to quantify judge-human agreement.

---

## 7. Larger and more difficult evaluation set

I would expand the evaluation set and deliberately include:

* ambiguous examples;
* multi-issue messages;
* short messages;
* noisy social-media messages;
* security-sensitive cases;
* rare intents.

This would provide a better estimate of real-world robustness.

---

# 15. Reproducibility

The repository contains:

```text
AppleSupport-AI-Agent/
│
├── README.md
├── requirements.txt
├── decision_log.md
├── run_pipeline.py
│
├── notebooks/
│   └── development.ipynb
│
├── src/
│   ├── __init__.py
│   ├── extract_apple.py
│   ├── build_intents.py
│   ├── retrieval.py
│   ├── agent.py
│   └── evaluation.py
│
├── evaluation/
│   ├── golden_set.csv
│   ├── results.csv
│   ├── retrieval_results.csv
│   ├── response_judge.csv
│   ├── agent_outputs.csv
│   └── confusion_matrix.png
│
└── report/
    └── report.md
```

The raw Twitter dataset is intentionally not committed to the repository.

The complete development and evaluation workflow is documented in:

```text
notebooks/development.ipynb
```

Dependencies are listed in:

```text
requirements.txt
```

The README provides instructions for reproducing the pipeline and evaluation.

---

# 16. Key Takeaways

This project demonstrates a lightweight support-agent architecture combining:

```text
Intent Classification
        +
Historical Retrieval
        +
Rule-Based Escalation
        +
Grounded Response Generation
```

The evaluation highlights an important distinction between components.

The current intent classifier achieves:

**28.5% accuracy and 0.2402 macro-F1.**

The retrieval component obtains:

**0.8864 mean top-1 cosine similarity.**

The escalation component obtains:

**0.3810 precision, 0.2286 recall, and 0.2857 F1.**

The results indicate that lexical similarity can retrieve highly overlapping historical cases while the manually defined intent boundaries remain difficult for a simple TF-IDF classifier.

The main improvement opportunity is therefore **semantic understanding and confidence-aware routing**, rather than simply increasing the number of keyword rules.
