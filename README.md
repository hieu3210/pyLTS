# pyLTS

pyLTS — Linguistic Time Series Forecasting with Hedge Algebras

Abstract
-------
pyLTS is a Python implementation of a linguistic time series forecasting framework grounded in Hedge Algebra (HA) theory. The project implements models that represent and reason about imprecise temporal data using linguistic variables, hedges (linguistic modifiers), and algebraic operations defined by HA. The approach is designed for domains where human-like, qualitative descriptions complement or replace purely numerical forecasting (e.g., socio-economic indicators, environmental monitoring, and expert-driven signals).

Key features
- Model linguistic time series using Hedge Algebras (HA).
- Support for configurable generators and hedges to express linguistic terms.
- Dataset utilities and example time series included for experimentation.
- Reference implementations of core algorithms and a PSO-based parameter tuning helper.

Algorithm overview
------------------
The framework encodes time series observations as linguistic terms drawn from an HA-defined lattice. Hedges (e.g., "Little", "Very") and generators (positive/negative direction) transform base terms to express nuanced qualitative states. Forecasting proceeds by:

1. Mapping numeric observations to linguistic terms via fuzzification rules.
2. Modeling temporal dependencies in the linguistic domain (rule induction, pattern matching, or algebraic combination).
3. Optionally optimizing model parameters (e.g., membership thresholds, hedge strengths) with PSO.
4. Defuzzifying predicted linguistic terms back to numeric forecasts where required.

Hedge Algebra (HA) background
----------------------------
Hedge Algebra provides an algebraic structure for linguistic variables: a finite set of primary terms (generators) and a family of hedges that modify term intensity. HA defines ordering and combination operations that preserve semantic relationships between terms, enabling rigorous reasoning with qualitative descriptors.

Usage
-----
Clone the repository and install dependencies (if any). Example usage patterns and dataset loaders are provided under `data/` and `LTS/` modules for exploration and prototyping.

Example (conceptual)
--------------------
1. Prepare or load a numeric time series.
2. Configure an HA (generators, hedges, membership/alpha thresholds).
3. Convert the numeric series to linguistic terms.
4. Train or infer linguistic transition rules.
5. Produce linguistic forecasts and defuzzify to numeric values.

Datasets
--------
This repository contains sample datasets for experimentation in `datasets/` and connectors in `data/` to standard time series (financial, meteorological, and synthetic chaotic systems).

Contributing and citation
------------------------
Contributions are welcome — please open issues or pull requests. If you use pyLTS in your research, please cite the repository and relevant Hedge Algebra literature.

License
-------
This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the included `LICENSE` file or https://www.gnu.org/licenses/gpl-3.0.en.html

Contact
-------
For questions or collaboration, open an issue or contact the authors.
E-mail: hieu3210@gmail.com