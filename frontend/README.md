# SatQuery View

👤 ANUSHKA — Frontend, Visualization & Evaluation

Role

Frontend UI + Results Visualization + Evidence Display + Evaluation Dashboard + Report Presentation

Anushka, your responsibility is to make SatQuery AI easy to use, visually understandable, and demo-ready.

You don't need to handle the hardest AI/model implementation. Your focus is taking the outputs from Tilak, Shreyas, Shreedhar, Shourya, and Vinayaka's backend and presenting them clearly to the user.

The goal is:

A judge should be able to upload satellite imagery, ask a natural-language question, and immediately understand what the system did, what it found, how confident it is, and what visual evidence supports the answer.

🎯 YOUR MAIN OBJECTIVE

Build this user experience:

User

 ↓

Upload Image / Images

 ↓

See Image + Metadata

 ↓

Ask Natural-Language Query

 ↓

SatQuery AI processes

 ↓

Show Execution Progress

 ↓

Show Final Answer

 ↓

Show Confidence

 ↓

Show Visual Evidence

 ↓

Show Change Map / Grounding / Optical-SAR Results

 ↓

Download Report

The PS expects an interactive GUI/web application connected to the agentic AI backend.

📌 YOUR MAIN RESPONSIBILITIES

You own:

1. Upload Interface

2. Query Interface

3. Image Viewer

4. Results Dashboard

5. Grounding Visualization

6. Change Map Visualization

7. Optical-SAR Comparison View

8. Confidence/Evidence Display

9. Execution Trace Display

10. Evaluation Dashboard

11. Downloadable Report UI

12. Demo/presentation polish

📂 FIRST — AUDIT THE CURRENT FRONTEND

Inspect the existing:

frontend/

Find the current:

UploadPanel

QueryBox

ImageViewer

ResultsDashboard

ExecutionTracePanel

and understand how they currently communicate with:

backend/api/routes.py

Do not rewrite the entire frontend immediately.

First determine:

What already works

What is incomplete

Which APIs are already available

What response format the frontend expects

What needs to change after the backend specialists integrate their models

🖼️ 1. UPLOAD INTERFACE

The UI should allow the user to upload:

Single image

GeoTIFF / TIFF

for:

VQA

captioning

grounding

Two images

For:

T1 + T2

for temporal/change analysis.

Optical + SAR

For:

Optical + SAR

for cross-modal analysis.

⚠️ SHOW VALIDATION ERRORS CLEARLY

If the user uploads the wrong number/type of images, don't just show:

Error 500

Show something understandable:

Two images are required for temporal comparison.

or:

This query requires an optical + SAR pair.

The actual validation decision comes from Vinayaka + Shourya's backend, but you present the result properly.

📝 2. QUERY BOX

The main UI should encourage natural-language questions.

Example:

Ask SatQuery AI...

"What is present in this image?"

Do not force the user to select:

☑ VQA

☑ Grounding

☑ Change Detection

The agent should determine the task automatically.

You can provide example prompts such as:

Describe this scene

Where are the buildings?

Is there a water body?

What changed between these images?

Compare the optical and SAR imagery

🔄 3. PROCESSING STATE

While the backend is processing, clearly show that the system is working.

Example:

Analyzing...

✓ Query understood

✓ Input validated

✓ Task identified

✓ Specialist selected

● Running analysis

○ Evidence fusion

○ Generating result

The backend should provide the actual execution information.

Do not fake completion steps just for animation.

🧠 4. FINAL RESULT DASHBOARD

The final result should be easy to understand.

Example:

━━━━━━━━━━━━━━━━━━━━━━

       SATQUERY AI

━━━━━━━━━━━━━━━━━━━━━━

Answer

"A water body is visible in the southern

portion of the image."

Confidence

████████░░ 87%

Evidence

• VLM analysis

• Image region

Model

[actual model]

━━━━━━━━━━━━━━━━━━━━━━

Keep the important answer visually prominent.

🗺️ 5. GROUNDING VISUALIZATION

When Tilak's grounding model returns:

bbox:

[x1, y1, x2, y2]

display the boxes directly over the image.

Example:

       SATELLITE IMAGE

      ┌──────────────┐

      │              │

      │   ┌─────┐    │

      │   │BUILD│    │

      │   └─────┘    │

      │        ┌───┐ │

      │        │ B │ │

      │        └───┘ │

      └──────────────┘

Query: "Where are the buildings?"

Show:

bounding boxes

labels

confidence

coordinates where useful

🔄 6. CHANGE VISUALIZATION

When Shreedhar's module returns a change map:

Show:

Before | After | Change

Preferably allow:

[ Before ]

[ After ]

[ Change Overlay ]

The user should be able to visually understand where the change occurred.

Also display:

Change detected: YES

Changed area: XX%

Confidence: XX

Only display values actually returned by the backend.

🛰️ 7. OPTICAL + SAR VIEW

For Shreyas's output, provide a clear comparison:

┌──────────────┬──────────────┐

│    OPTICAL   │     SAR      │

│              │              │

│    IMAGE     │    IMAGE     │

│              │              │

└──────────────┴──────────────┘

OPTICAL EVIDENCE

...

SAR EVIDENCE

...

COMPLEMENTARY INFORMATION

...

This is important for the SIH demonstration because judges should immediately understand that the system is actually using both modalities.

🔍 8. EVIDENCE DISPLAY

Don't show only:

Answer: Urban development increased.

Show why.

Example:

Answer

Urban development increased in the eastern region.

Evidence

✓ Optical evidence

  New visible structures detected

✓ Temporal evidence

  Difference detected between T1 and T2

✓ Spatial evidence

  Change concentrated in eastern region

The exact evidence comes from the backend.

📊 9. CONFIDENCE DISPLAY

Display confidence clearly but don't make it look scientifically stronger than it is.

For example:

Confidence: 84%

or:

Confidence

0.84

Also support an uncertainty/disagreement indicator:

⚠ Evidence disagreement detected

if Vinayaka's evidence-fusion layer returns that.

Do not calculate your own fake confidence in the frontend.

🧾 10. EXECUTION TRACE

The PS specifically requires an observable/auditable execution summary.

Create a clean UI for it.

Example:

Execution Summary

Query

"What changed?"

Input

2 GeoTIFF images

Detected task

Bi-temporal Change Analysis

Validation

✓ Compatible images

Specialist

Change Intelligence

Processing

✓ Registration

✓ Change detection

✓ Change analysis

✓ Evidence fusion

Result

Change detected

Confidence

0.86

This is not chain-of-thought.

Do not display hidden reasoning.

Only display the observable execution information supplied by the backend.

📄 11. DOWNLOADABLE REPORT

The user should eventually be able to click:

Download Report

and obtain a report containing relevant:

Query

Input information

Detected task

Model information

Answer

Confidence

Evidence

Grounding regions

Change map

Optical-SAR findings

Execution summary

Coordinate with Vinayaka regarding the backend report endpoint.

The project plan specifically includes a downloadable report as part of the final polish/evaluation phase. �

SatQuery-AI-All-Phases.pdf

📈 12. EVALUATION DASHBOARD

You also own the presentation of evaluation results.

Create a simple dashboard where we can eventually display actual benchmark results:

Evaluation

VQA

Accuracy / appropriate metric: XX

Grounding

IoU / mIoU: XX

Captioning

Metric: XX

Change-VQA

Metric: XX

IMPORTANT

Do not enter the existing README's claimed scores as if they are verified.

The current repository's AI implementations contain placeholders, so those numbers need to be regenerated after the real models are implemented.

Only display actual experiment results supplied by the respective members.

🎨 UI DESIGN GOAL

Keep it:

Professional

Clean

Space/remote-sensing themed

Easy to understand

Demo-friendly

Don't overload the screen.

The judge should understand the workflow in seconds.

🧩 COMPONENT STRUCTURE

You can organize the frontend approximately as:

frontend/

│

├── components/

│   ├── UploadPanel

│   ├── QueryBox

│   ├── ImageViewer

│   ├── MetadataPanel

│   ├── ResultsDashboard

│   ├── GroundingOverlay

│   ├── ChangeMapViewer

│   ├── OpticalSARViewer

│   ├── ConfidenceCard

│   ├── EvidencePanel

│   ├── ExecutionTracePanel

│   └── ReportDownload

│

├── services/

│   └── api

│

└── ...

Adapt this to the existing frontend architecture instead of blindly recreating it.

🤝 WHO YOU NEED TO COORDINATE WITH

Vinayaka

This is your main backend integration contact.

You need from him:

API endpoints

request schema

response schema

execution trace format

error format

report endpoint

Tilak

Need:

VQA response

caption response

grounding response

bounding boxes

confidence

evidence

Shreyas

Need:

optical evidence

SAR evidence

complementary findings

confidence

Shreedhar

Need:

change map

change percentage

changed regions

change description

confidence

Shourya

Need:

metadata

modality

CRS

dimensions

bands

resolution

validation status

🚫 WHAT YOU SHOULD NOT DO

Do NOT implement:

VLM

LoRA

VQA logic

Change detection

SAR analysis

Agent planner

Geospatial algorithms

Your job is to visualize and integrate their outputs.

🚫 VERY IMPORTANT — DON'T FAKE UI RESULTS

Do not create UI like:

Accuracy: 94.2%

Confidence: 97%

Change: 28%

just because the demo needs numbers.

If the backend hasn't produced the value:

Not available

or don't display it.

A beautiful fake result is worse than a simple truthful UI in an SIH evaluation.

🧪 TESTING REQUIREMENTS

Test:

Upload

Single image

Two temporal images

Optical + SAR

Invalid file

Query

VQA

Caption

Grounding

Temporal

Optical-SAR

Results

Verify correct rendering of:

Answer

Confidence

Evidence

Bounding boxes

Change map

Metadata

Execution trace

Error handling

Backend error should become a human-readable frontend message.

Responsive UI

Test:

Laptop

Desktop

Different browser widths

🏆 DEFINITION OF DONE

Anushka's work is complete when a judge can do:

Upload

  ↓

Ask

  ↓

Watch processing

  ↓

See answer

  ↓

See evidence

  ↓

See visualization

  ↓

Understand confidence

  ↓

Inspect execution summary

  ↓

Download report

without needing a developer to explain what happened.

📢 WHAT TO REPORT TO VINAYAKA

After every major milestone:

1. What I implemented

2. Files changed

3. Components created/modified

4. APIs integrated

5. Upload functionality

6. Query functionality

7. Result visualization

8. Grounding visualization

9. Change-map visualization

10. Optical-SAR visualization

11. Execution trace

12. Confidence/evidence display

13. Report download

14. Evaluation dashboard

15. Tests performed

16. UI issues

17. Backend dependencies

18. What I need from Vinayaka

19. What I need from other members

20. Git commit/hash

🤖 COPY-PASTE PROMPT FOR ANTIGRAVITY / CLAUDE CODE / CODEX / COPILOT

You are working on SIH 2026 Problem Statement 26167:

"SatQuery AI - An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries."

Your responsibility is:

FRONTEND + VISUALIZATION + EVALUATION

FIRST AUDIT THE EXISTING FRONTEND.

Inspect:

frontend/

backend/api/routes.py

Identify existing:

- UploadPanel

- QueryBox

- ImageViewer

- ResultsDashboard

- ExecutionTracePanel

- API service code

- report/download functionality

DO NOT rewrite the entire frontend before understanding the current implementation.

MAIN GOAL:

Create a professional interactive GUI for SatQuery AI.

The user should be able to:

1. Upload a single remote-sensing image.

2. Upload a bi-temporal pair.

3. Upload an optical + SAR pair.

4. Enter a natural-language query.

5. Submit the query.

6. See actual processing status.

7. See the final answer.

8. See confidence.

9. See supporting evidence.

10. See visual outputs.

11. See execution summary.

12. Download a report.

SINGLE IMAGE UI:

Support:

- VQA

- captioning

- grounding

GROUNDING:

When backend returns bounding boxes, overlay them on the original image.

Display:

- region

- label/query

- confidence

Do NOT generate fake bounding boxes in the frontend.

BI-TEMPORAL:

Display:

- T1

- T2

- change map

- change percentage if returned

- changed regions

- change description

- confidence

Do NOT calculate fake change results in the frontend.

OPTICAL + SAR:

Display:

- optical image

- SAR image

- optical evidence

- SAR evidence

- complementary findings

- fused answer

- confidence

Do not simply concatenate text if the backend provides structured evidence.

METADATA:

Display useful metadata returned by the backend:

- modality

- dimensions

- bands

- CRS

- resolution

- file type

- validation status

Do not invent metadata.

EXECUTION TRACE:

Create a clean observable execution summary containing:

- query

- inputs

- detected task

- validation result

- selected specialist

- model name/version if provided

- processing steps

- result

- confidence

- evidence

- errors/disagreements

Do NOT display chain-of-thought or hidden reasoning.

ERROR HANDLING:

Convert backend errors into clear user messages.

Examples:

- missing second image

- unsupported format

- incompatible images

- model unavailable

- processing failure

EVALUATION:

Create a dashboard that can display actual benchmark results for:

- VQA

- captioning

- grounding

- change-VQA

- other evaluated modules

Do NOT hard-code evaluation scores.

The current README contains claimed metrics that may not be trustworthy because some current backend modules are placeholders. Only display metrics supplied after genuine evaluation.

REPORT:

Provide UI support for downloadable reports using the actual backend report endpoint.

Report may include:

- query

- inputs

- task

- model

- answer

- confidence

- evidence

- grounding regions

- change map

- optical-SAR findings

- execution summary

TEAM INTEGRATION:

Vinayaka:

Agentic AI + backend integration.

Get API contracts and response schemas from him.

Tilak:

VLM/VQA/captioning/grounding.

Use his actual output schema.

Shreyas:

Optical-SAR.

Use his structured optical/SAR/fusion results.

Shreedhar:

Change intelligence.

Use his change map/statistics/description.

Shourya:

Geospatial metadata and validation.

Display returned metadata.

Do NOT implement their core AI algorithms.

IMPORTANT:

The frontend must visualize actual backend outputs.

NEVER create fake:

- confidence

- accuracy

- VQA answers

- grounding boxes

- change maps

- optical/SAR results

- model names

- evaluation scores

WORK ORDER:

Phase 1:

Audit existing frontend.

Phase 2:

Document current components and API contracts.

Phase 3:

Improve upload workflow.

Phase 4:

Improve query workflow.

Phase 5:

Build results dashboard.

Phase 6:

Build grounding overlay.

Phase 7:

Build temporal/change visualization.

Phase 8:

Build optical-SAR comparison.

Phase 9:

Build evidence/confidence display.

Phase 10:

Build execution trace.

Phase 11:

Build report/download UI.

Phase 12:

Build evaluation dashboard.

Phase 13:

Run end-to-end tests.

At the end report:

- files changed

- components created/modified

- APIs integrated

- screenshots/test results if available

- supported workflows

- known limitations

- dependencies on Vinayaka

- dependencies on Tilak/Shreyas/Shreedhar/Shourya

- git commit/hash

Do not claim functionality is complete unless it actually works.

🔴 Anushka's one-line mission

“Turn the real AI work of the team into a clean, visual, judge-friendly SatQuery AI experience where every answer, map, confidence score, and piece of evidence is understandable and traceable.”
i want to do this.. do you have any qns?

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/9e4a68f7-94b2-41fb-b6b2-9468364a6e49).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
