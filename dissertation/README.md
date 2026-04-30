# COMP390 Dissertation -- LaTeX Template

University of Liverpool -- BSc / MEng Computer Science Honours Year Project.
Multi-file LaTeX project skeleton, anonymous-after-frontmatter page styles,
biblatex (Harvard default, IEEE alternative).

## Compile

This template uses **tectonic** (a self-contained TeX engine with
on-demand package downloads). It is the simplest way to compile on macOS
and avoids the multi-gigabyte MacTeX install.

Install once:

```bash
brew install tectonic
```

Compile:

```bash
tectonic -X compile main.tex
```

Tectonic detects the bibliography automatically, runs its internal BibTeX,
and re-runs LaTeX until cross-references converge. Output is `main.pdf`.

### Alternative toolchain (full TeX Live)

If you prefer a traditional install:

```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install latexmk
latexmk -pdf main.tex
```

The default bibliography uses **`agsm` (Harvard) via `natbib + bibtex`**, so
no external `biber` is needed regardless of which engine you pick.

## Where to edit personal information

All identifying information lives in **four `\newcommand` lines at the top
of `main.tex`**:

```latex
\newcommand{\studentname}{Qilin Zheng}
\newcommand{\studentid}{201600000}
\newcommand{\projecttitle}{...}
\newcommand{\degree}{BSc Computer Science}
```

Plus two further lines for supervisor / assessor / submission date.

These commands are referenced **only on the personal pages** (`title_page.tex`,
`dedication.tex`, `acknowledgements.tex`). The rest of the document is fully
anonymous: page headers / footers from the Anonymous Title Page onward show
only the chapter title and page number, never name or student ID. This means
a marker can detach the personal pages cleanly.

## How to switch citation style

Default is Harvard author-year (`agsm` BibTeX style). Two changes flip it
to IEEE numeric:

1. In `main.tex`, replace
   ```latex
   \usepackage[round, authoryear]{natbib}
   ```
   with
   ```latex
   \usepackage[numbers, sort&compress]{natbib}
   ```

2. In `chapters/references.tex`, replace
   ```latex
   \bibliographystyle{agsm}
   ```
   with
   ```latex
   \bibliographystyle{IEEEtran}
   ```

Other BibTeX styles available out of the box:

| Style                            | `\bibliographystyle{}` argument |
|----------------------------------|---------------------------------|
| Harvard (default)                | `agsm`                          |
| APA-like                         | `apalike`                       |
| Chicago author-year              | `chicago`                       |
| IEEE numeric                     | `IEEEtran`                      |
| ACM numeric                      | `acm`                           |
| Plain numeric                    | `plain`                         |

After switching, just run `tectonic -X compile main.tex` again.

## Word count

The handbook expects 8,000--10,000 words for the dissertation body
(excluding code, figure captions, references, appendices). To estimate:

```bash
texcount main.tex -inc -sum -total
```

Use `-merge` to follow `\include`/`\input` recursively. Use `-1` to print
a single-line total.

## Project structure

```
dissertation/
|-- main.tex                  one entry point; \include's chapters
|-- references.bib            bibliography (biblatex format)
|-- README.md                 this file
|-- chapters/
|   |-- title_page.tex
|   |-- dedication.tex
|   |-- acknowledgements.tex
|   |-- anonymous_title.tex
|   |-- abstract.tex
|   |-- ethical_compliance.tex
|   |-- toc.tex
|   |-- introduction.tex
|   |-- requirements_analysis.tex   (optional -- delete if no users)
|   |-- design.tex
|   |-- implementation.tex
|   |-- testing_evaluation.tex
|   |-- project_ethics.tex
|   |-- conclusion.tex
|   |-- bcs_reflection.tex
|   |-- references.tex
|   `-- appendices.tex
`-- figures/                  put .pdf / .png / .jpg figures here
```

## Page-style design

Two page styles are defined in `main.tex`:

* **`personal`** -- empty header and footer. Used for `title_page.tex`,
  `dedication.tex`, `acknowledgements.tex`. The student name and ID
  appear ON these pages but never in headers or footers, so a marker can
  remove these three pages cleanly.

* **`anonymous`** -- chapter title on the left, page number on the right.
  Active from `anonymous_title.tex` to the end of the document. Contains
  no identifying information.

Page numbering:

* Personal pages: hidden (`\pagenumbering{gobble}`)
* Anonymous front matter: lower-case Roman (`i, ii, iii, ...`)
* Main chapters onward: Arabic, restarting at 1

## TODO notes

`\todo{...}` from the `todonotes` package shows margin annotations during
drafting. A consolidated list of all open notes is printed at the end of
the document via `\listoftodos`. Comment out that line in `main.tex`
before final submission.

## Code listings

The default code environment uses `listings`. Either of the following works:

```latex
\begin{lstlisting}[language=Python, caption=Example, label=lst:foo]
def hello():
    print("hello")
\end{lstlisting}
```

```latex
\begin{code}[language=Python, caption=Example, label=lst:foo]
def hello():
    print("hello")
\end{code}
```

To switch to `minted` (syntax highlighting via Pygments) you will need
Python and to compile with `--shell-escape`. The default `listings`-based
setup needs neither.

## Common issues

* **Missing references / `??` in citations.** Run `biber main` between
  `pdflatex` passes, or just use `latexmk -pdf`.
* **Headers still show name/ID on anonymous pages.** Check that no
  chapter file uses `\thispagestyle{personal}` past `anonymous_title.tex`.
* **Figures not found.** Drop them in `figures/`; `\graphicspath` is
  pre-configured.

## Submission checklist (handbook-aligned)

- [ ] Title page lists module code (COMP390), title, name, ID, supervisor.
- [ ] Anonymous title page contains no name / no ID.
- [ ] Statement of Ethical Compliance has data category and participant
      category filled in.
- [ ] Abstract is at most one page.
- [ ] Aims and Requirements are numbered (R1, R2, ...).
- [ ] Evaluation chapter contains measurements, not self-reflection.
- [ ] BCS chapter contains a one-page+ first-person self-reflection.
- [ ] Word count is 8,000--10,000 (`texcount main.tex -inc -sum`).
- [ ] `\listoftodos` is commented out for the final PDF.
- [ ] `latexmk -C; latexmk -pdf main.tex` rebuilds cleanly.
