# CPBM algorithm
(Capital-Pulse Behavioral Model) A conceptual model for tracking the pulse of local capital to predict purchasing behavior across demographics Central Idea Instead of categorizing individuals into fixed segments, we track the movement of capital as a dynamic phenomenon. Each individual is a unit of financial activity that pulsates to its own rhythm
# Capital-Pulse Behavioral Model (CPBM)

**A four-layer algorithmic framework for predicting purchase-trend diffusion
across socioeconomic strata via individual capital-flow signatures.**

ORCID: [0009-0000-5475-3970](https://orcid.org/0009-0000-5475-3970)
OSF: [osf.io/9ukxe](https://osf.io/9ukxe/)
Paper: see `docs/cpbm_paper.pdf` [https://sciprofiles.com/profile/ahmed_elsayed](https://sciprofiles.com/profile/ahmed_elsayed)

---

## Installation

```bash
git clone https://github.com/ahmed-elsayed-99/CPBM-algorithm.git
cd cpbm
pip install -e .
```

## Quick Start

```python
from cpbm.data.synthetic import SyntheticCommunity
from cpbm.core.signature import SignatureExtractor
from cpbm.core.diffusion import DiffusionLayer
from cpbm.core.stratum import StratumGradient
from cpbm.models.ensemble import CPBMEnsemble

community = SyntheticCommunity(n=500, seed=42).generate()
extractor = SignatureExtractor(window_days=90)
Phi = extractor.fit_transform(community["transactions"])

diffusion = DiffusionLayer()
diffusion.fit(community["adoption_history"])

stratum = StratumGradient()
tau = stratum.fit(community["stratum_adoption"])

model = CPBMEnsemble()
model.fit(Phi, community["labels"],
          tau=tau, diffusion_params=diffusion.params_)

probas = model.predict_proba(Phi)
print(f"AUC: {model.evaluate(Phi, community['labels'])['auc']:.4f}")
```

## Architecture

```
Layer 1  →  Individual Pulse Signature  φᵢ ∈ ℝ⁷
Layer 2  →  Social Diffusion PDE        ∂P/∂t = D∇²P + αS - βP + γN
Layer 3  →  Stratum-Gradient Bass       τ = t*_lower - t†_upper
Layer 4  →  Hybrid Ensemble             XGBoost + LSTM + GAT
```

## Citation

```bibtex
@misc{elsayed2025cpbm,
  author       = {Elsayed, Ahmed},
  title        = {Capital-Pulse Behavioral Model (CPBM)},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ahmed-elsayed-99/CPBM-algorithm}},
  note         = {OSF: https://osf.io/9ukxe/}
}
```
