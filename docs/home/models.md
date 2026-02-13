<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Open-Source Models Based on PyPSA

PyPSA is the software framework to build open-source energy system models. Many
model implementations by different [users](users.md) already exist. Some are
open source and actively maintained. Below is a list of some known models.

## Regional Models

### Global
- **[:earth_africa: PyPSA-Earth](https://github.com/pypsa-meets-earth/pypsa-earth)**: Global energy system model, maintained by [pypsa-meets-earth](https://pypsa-meets-earth.github.io/)

### Europe
- **[:flag_eu: PyPSA-Eur](https://github.com/PyPSA/pypsa-eur)**: European energy system model, maintained by [TU Berlin](https://www.tu.berlin/en/ensys)
- **[:flag_de: PyPSA-DE](https://github.com/PyPSA/pypsa-de)**: German energy system model, maintained by [TU Berlin](https://www.tu.berlin/en/ensys)
- **[:earth_africa: PyPSA-CEE](https://github.com/ember-climate/pypsa-cee)**: Central and Eastern Europe power system model, maintained by [Ember](https://ember-energy.org/)
- **[:flag_pl: PyPSA-PL](https://github.com/instrat-pl/pypsa-pl)**: Polish power system model, developed by [Instrat.pl](https://instrat.pl/en/)
- **[:flag_es: PyPSA-Spain](https://github.com/cristobal-GC/pypsa-spain)**: Spanish energy system model, maintained by Polytechnic University of Madrid
- **[:flag_gb: PyPSA-UK](https://github.com/ember-climate/pypsa-uk)**: United Kingdom power system model, maintained by [Ember](https://ember-energy.org/)
- **[:flag_gb: PyPSA-GB](https://github.com/andrewlyden/PyPSA-GB)**: Great Britain power system model, maintained by the University of Edinburgh
- **[:flag_gb: PyPSA-FES](https://github.com/centrefornetzero/pypsa-fes)**: Great Britain power system model, maintained by Octopus Energy's [Centre for Net Zero](https://www.centrefornetzero.org)
- **[:flag_at: PyPSA-AT](https://github.com/AGGM-AG/pypsa-at)**: Austrian power and gas system model, maintained by [Austrian Gas Grid Management AG (AGGM)](https://www.aggm.at/en/)
- **[:flag_eu: PyPSA-IEI](https://github.com/Fraunhofer-IEG/PyPSA-IEI)**: European energy system model, maintained by [Fraunhofer IEG](https://www.ieg.fraunhofer.de/en.html), [Fraunhofer ISI](https://www.isi.fraunhofer.de/en.html), and [d-fine](https://www.d-fine.com/en/)
- **[:flag_nl: PyPSA-NL2025](https://github.com/Tanneheemsbergen/pypsa-NL2025)**: Dutch energy system model simulating day-ahead, imbalance-only, and value stacking for energy storage projects, maintained by Tanne Heemsbergen

### Asia
- **[:flag_cn: PyPSA-China-PIK](https://github.com/pik-piam/PyPSA-China-PIK)**: Chinese power and heat sector-coupling model, maintained by [PIK](https://www.pik-potsdam.de) based on a [previous version](https://github.com/Xiaowei-Z/PyPSA-China)
- **[:flag_vn: PyPSA-VN](https://github.com/fiasresna/pypsa-vn)**: Vietnamese electricity model, developed by FIAS
- **[:flag_kr: PyPSA-KR](https://energyinnovation.korea.ac.kr/research/code-and-data)**: South Korean energy system model, developed by Korea University
- **[:flag_jp: PyPSA-Japan](https://github.com/smdumlao/demandfingerprint/tree/main/papers/coaldecommissioning)**: Japanese power system model, developed by Samuel Matthew Dumlao
- **[:flag_kz: PyPSA-Kazakhstan](https://github.com/pypsa-meets-earth/pypsa-kz-data)**: Kazakhstan power system model, developed by [Open Energy Transition ](https://www.openenergytransition.org/)
- **[:earth_asia: TZ-APG](https://www.transitionzero.org/products/tz-asean-power-grid-model)**: ASEAN power system model, maintained by [TransitionZero](https://www.transitionzero.org/)
- **[:earth_asia: PyPSA-ASEAN](https://github.com/pypsa-meets-earth/pypsa-asean)**: ASEAN power system model, developed by [pypsa-meets-earth](https://pypsa-meets-earth.github.io/)

### Americas
- **[:flag_us: PyPSA-USA](https://github.com/pypsa/pypsa-usa)**: United States energy system model, maintained by Stanford University
- **[:flag_br: PyPSA-Brazil](https://gitlab.com/dlr-ve/esy/open-brazil-energy-data/open-brazilian-energy-data)**: Brazilian power system model (see [paper](https://doi.org/10.1038/s41597-023-01992-9)), developed by the German Aerospace Center (DLR)
- **[:flag_ca: PyPSA-BC](https://github.com/DeltaE/PyPSA_BC)**: BC PyPSA work for the PICS Decarbonization project, maintained by Simon Fraser University

### Africa
- **[:flag_za: PyPSA-RSA](https://github.com/MeridianEconomics/pypsa-rsa)**: South-African electricity model, maintained by [Meridian Economics](https://meridianeconomics.co.za/)
- **[:flag_za: PyPSA-ZA](https://github.com/PyPSA/pypsa-za)**: Previous version of PyPSA-RSA

### Oceania
- **[:flag_au: ISPyPSA](https://github.com/Open-ISP/ISPyPSA)**: Australian capacity expansion model, maintained by [CEEM University of New South Wales](https://ceem.unsw.edu.au)
- **[:flag_nz: PyPSA-NZ](https://github.com/energyLS/pypsa-nz)**: New Zealand energy system model, developed by Leon Schumm


## Interactive Tools and Specialized Applications

- Build your own global zero emission scenario [model.energy](https://model.energy), maintained by [pypsa.org](https://pypsa.org)
    - Define your own sector-coupled European scenarios [scenarios](https://model.energy/scenarios/)
    - Build your own green energy import supply chain [supply chain](https://model.energy/green-energy-imports/)
    - See how the [future](https://model.energy/future/) German energy system might operate with today's weather
- Transport of chemical energy carriers to Germany [TRACE](https://github.com/euronion/trace)
- Transmission grid optimisation [eTraGo](https://github.com/openego/eTraGo)
- Distribution grid optimisation [dDisGo](https://github.com/openego/eDisGo)
- Assessing national mid-/long-term energy scenarios using a least-cost, multi-sectoral optimisation approach [PyPSA-SPICE](https://github.com/agoenergy/pypsa-spice) and visualization, maintained by [Agora Energiewende](https://github.com/agoenergy)

!!! tip "Add your model to the list"
    You know another open-source model based on PyPSA or are developing one? Please reach out to us via [GitHub](https://github.com/PyPSA/PyPSA) or [Discord](https://discord.gg/AnuJBk23FU) and we will add it to the list!
