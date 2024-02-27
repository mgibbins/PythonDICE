# Perform model runs and output plots used in the README

import sys

sys.path.append("../PythonDICE")

from DICE import DICE
from PlotResults import plot_results

# run models
DICE_2016 = DICE()
DICE_updated_parameters = DICE(use_updated_parameters=True)
TCRE_DICE = DICE(
    use_updated_parameters=True, user_parameters={"climate_module": "TCRE"}
)

# plot and save results
fig1 = plot_results(
    models={DICE_2016: "DICE 2016", DICE_updated_parameters: "updated parameters"},
    plot_to_year=2300,
)
fig1.savefig("images/DICE_2016_vs_updated_parameters.svg", bbox_inches="tight")

fig2 = plot_results(
    models={DICE_updated_parameters: "original climate", TCRE_DICE: "TCRE climate"},
    plot_to_year=2300,
)
fig2.savefig("images/DICE_vs_TCRE_climate.svg", bbox_inches="tight")
