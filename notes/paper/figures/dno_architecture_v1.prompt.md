# DNO architecture figure

Generated with the built-in image-generation tool, based on
`models/dno-net/dno_net_v2.py` and the active C27 configuration.

## Caption

The neural Dirichlet–Neumann operator adds a learned correction to the fixed
Craig–Sulem terms G0 and G1. Each of B = 2,560 parallel branches applies a
depth-dependent real Fourier filter to xi, multiplies by a surface-dependent
spatial weight, and applies the same filter again. Branches share a surface
feature network and are arranged in eight groups of 320; each group has its
own multiplier network. The summed correction is scaled by 1/sqrt(B).
The bias-free surface network and quadratic lift make the correction start at
second order in eta; zero-initialized spatial-weight heads make the initial
model exactly G0 + G1.
For fixed surface and depth the operator is linear in xi and self-adjoint;
positivity is not imposed.

The drawing suppresses normalization and depth clipping: surface features use
normalized eta, model depth is capped at 5, and the displayed equation represents
physical output before any evaluator mean projection.

## Generation prompt

Use case: scientific-educational / infographic-diagram.
Create a polished, publication-quality neural-operator architecture figure for a scientific journal. Landscape canvas approximately 3600 x 2200 pixels, pure white background, crisp flat vector-like drawing, generous whitespace, consistent thin strokes, restrained navy/teal/ochre palette that also reads in grayscale. No gradients, shadows, 3D, decorative art, logos, watermark, or poster-like headline. Beautiful compact sans-serif labels and correctly typeset mathematical symbols with subscripts. All text must be easily legible when placed across two journal columns.

The figure has two vertically arranged panels, labelled "(a) Operator overview" and "(b) One learned branch". Thin panel separator, aligned margins. This is an accurate computational graph, not an illustrative analogy. Follow the exact arrow topology below. Arrowheads show direction; solid arrows carry fields and dashed arrows supply learned filter coefficients.

PANEL (a), upper third:
At far left an input box with three lines "Surface η", "Potential ξ", "Depth h". A single input bundle branches into TWO paths, so all three inputs are visibly available to BOTH paths.
Upper path: a light neutral-gray block labelled "Fixed analytic baseline" with formula "(G₀ + G₁)ξ".
Lower path: a pale teal block labelled "Parallel learned branches" with small second line "B = 2,560  (8 groups × 320)" followed by a separate compact block labelled "Sum / √B".
Both paths terminate at a single circular plus node to the right, then a rightward arrow to "Predicted Gξ".
Do NOT draw the analytic output feeding the learned branches; they operate in parallel on the inputs. Do NOT depict 8 sequential network layers.

PANEL (b), lower larger panel:
A detailed schematic of ONE generic learned branch b.
Its central horizontal data path is EXACTLY:
"ξ" -> blue outlined box "Fourier filter M_b" -> circular multiplication node "×" -> second identical blue outlined box "Fourier filter M_b" -> "r_b".
Directly beneath the multiplication node label "Physical-space product". Directly beneath the two filter boxes print the smaller definitions "F⁻¹[m_b · F(·)]" on BOTH. Keep these definitions well spaced from the main arrow.
Above this path: "η" -> pale teal rectangle "Surface features + shared network" -> small teal box "Spatial weight a_b(η)" -> a vertical solid arrow that enters the multiplication node from above. The spatial-weight field must NOT feed either Fourier filter.
Below the central path, well separated: "(k, h)" -> pale ochre rectangle "Multiplier network" -> "Real coefficients m_b(k,h)". A dashed coefficient bus branches upward to the bottom of BOTH Fourier filter boxes, never to the multiplication node. Label the bus "Same filter on both sides". Make control arrows unambiguous, using clean orthogonal routing without tangles.
The input to the multiplier network is ONLY physical wavenumber k and water depth h, NOT eta or xi.
Note under the branch at the lower left, unobtrusively: "Shown once; evaluated in parallel for b = 1, …, B."

At the bottom, separated by whitespace, typeset this exact summary equation in a single readable centered line:
"Gθ(η,h)ξ = (G₀ + G₁)ξ + B⁻¹ᐟ² ∑_{b=1}^{B} M_b(h)[a_b(η) M_b(h)ξ]"
Use genuine standard mathematical typesetting for the inverse square root, summation limits, theta subscript and branch subscripts, not literal markup.
A small final line: "Linear in ξ · Nonlinear in η · Self-adjoint for fixed η and h".
Avoid claiming positive definiteness, positivity, conservation, translation-equivariance, independent networks per branch, or any experimental performance result.
Omit implementation-only input/output scaling, parameter count, training losses, and depth clipping from the drawing itself. This diagram is the physical-unit operator before any evaluator projection.
Keep text concise: use only the specified labels, formulas and panel headings. No additional caption paragraph inside the image. Before finalizing, ensure BOTH repeated M_b filters receive the same dashed coefficient input, and the η path enters ONLY the physical multiplication in panel (b).
