# CPG44 Mid-Semester Submission 2026

This folder is separate from the approved proposal and does not overwrite it. It contains the editable mid-semester report, a 15-slide presentation, simple PlantUML sources, rendered diagrams, and fixed image placeholders.

The editable LaTeX and PlantUML sources describe the relay-only wearable path.
Rendered diagrams and PDFs should be rebuilt after the final screenshots are
inserted.

## Main files

- `report/main.tex`: report entry point
- `presentation/presentation.tex`: 15-slide, 16:9 presentation
- `presentation/speaker_notes.md`: suggested 9 minute 50 second team handover
- `diagrams/*.puml`: editable UML sources
- `figures/*.png`: rendered UML and selected project results
- `build/CPG44_Mid_Semester_Report_2026.pdf`: final compiled report
- `build/CPG44_Mid_Semester_Presentation_2026.pdf`: final compiled slides

## Insert the final photos and screens

Place the following files in `figures/`:

- `dashboard_live.png`
- `dashboard_analytics.png`
- `dashboard_wearable.png`
- `hardware_front.jpg`

The report contains labelled positions for these four files. Do not change the filenames unless the matching LaTeX call is also changed.

## Build in WSL

From this folder, run:

```bash
./render_diagrams.sh
./build.sh
```

The build uses the `soccer` conda environment and Tectonic. The report copies the 12-point article class, 1.5 line spacing, default LaTeX typeface, and margins from the approved proposal source. The slides keep their own presentation typography.

## Before mentor approval

1. Insert the three final screens and the labelled hardware photograph.
2. Replace budget estimates with invoice values if available.
3. Add signatures, mentor approval, and the plagiarism report.
4. Run the live demo once using the order in Chapter 4.
5. Check that every numerical result used in the presentation is still present in the named run log.

The literature section is written as an original summary of the cited papers. It does not copy paper abstracts or claim their results as project results. Keep citations when editing technical claims.
